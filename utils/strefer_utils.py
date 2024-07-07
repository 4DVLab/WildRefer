import numpy as np
import cv2
import math
from scipy.spatial import Delaunay

cv2.ocl.setUseOpenCL(False)   
cv2.setNumThreads(0)

def load_image(img_filename):
    img = cv2.imread(img_filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    return img

def norm(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1*heading_angle)
    l,w,h = size
    l /= 2
    w /= 2
    h /= 2
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def batch_extract_pc_in_box3d(pc, boxes3d, sample_points_num, dim=4):
    objects_pc = []
    for i in range(len(boxes3d)):
        box3d = my_compute_box_3d(boxes3d[i][0:3], boxes3d[i][3:6], boxes3d[i][6])
        obj_pc, pc_ind = extract_pc_in_box3d(pc.copy(), box3d)
        if obj_pc.shape[0] == 0:
            obj_pc = np.zeros((sample_points_num, dim))
        else:
            obj_pc = random_sampling(obj_pc, sample_points_num)
        objects_pc.append(obj_pc)
    if len(objects_pc) > 0:
        objects_pc = np.stack(objects_pc, axis=0)
    else:
        objects_pc = np.zeros((0, sample_points_num, dim), dtype=np.float32)
    return objects_pc



def cal_corner_after_rotation(corner, center, r):
    x1, y1 = corner
    x0, y0 = center
    x2 = math.cos(r) * (x1 - x0) - math.sin(r) * (y1 - y0) + x0
    y2 = math.sin(r) * (x1 - x0) + math.cos(r) * (y1 - y0) + y0
    return x2, y2

def eight_points(center, size, rotation=0):
    x, y, z = center
    w, l, h = size
    w = w/2
    l = l/2
    h = h/2

    x1, y1, z1 = x-w, y-l, z+h
    x2, y2, z2 = x+w, y-l, z+h
    x3, y3, z3 = x+w, y-l, z-h
    x4, y4, z4 = x-w, y-l, z-h
    x5, y5, z5 = x-w, y+l, z+h
    x6, y6, z6 = x+w, y+l, z+h
    x7, y7, z7 = x+w, y+l, z-h
    x8, y8, z8 = x-w, y+l, z-h

    if rotation != 0:
        x1, y1 = cal_corner_after_rotation(corner=(x1, y1), center=(x, y), r=rotation)
        x2, y2 = cal_corner_after_rotation(corner=(x2, y2), center=(x, y), r=rotation)
        x3, y3 = cal_corner_after_rotation(corner=(x3, y3), center=(x, y), r=rotation)
        x4, y4 = cal_corner_after_rotation(corner=(x4, y4), center=(x, y), r=rotation)
        x5, y5 = cal_corner_after_rotation(corner=(x5, y5), center=(x, y), r=rotation)
        x6, y6 = cal_corner_after_rotation(corner=(x6, y6), center=(x, y), r=rotation)
        x7, y7 = cal_corner_after_rotation(corner=(x7, y7), center=(x, y), r=rotation)
        x8, y8 = cal_corner_after_rotation(corner=(x8, y8), center=(x, y), r=rotation)

    conern1 = np.array([x1, y1, z1])
    conern2 = np.array([x2, y2, z2])
    conern3 = np.array([x3, y3, z3])
    conern4 = np.array([x4, y4, z4])
    conern5 = np.array([x5, y5, z5])
    conern6 = np.array([x6, y6, z6])
    conern7 = np.array([x7, y7, z7])
    conern8 = np.array([x8, y8, z8])
    
    eight_corners = np.stack([conern1, conern2, conern3, conern4, conern5, conern6, conern7, conern8], axis=0)
    return eight_corners

def loc_pc2img(points, ex_matrix, in_matrix):
    assert len(points.shape) == 1 and len(points) > 6 
    x, y, z, w, l, h, r = points[:7]
    center = [x, y, z]
    size = [w, l, h]
    points = eight_points(center, size, r)
    points = np.insert(points, 3, values=1, axis=1)
    points_T = np.transpose(points)
    points_T[3, :] = 1.0

    # lidar2camera
    points_T_camera = np.dot(ex_matrix, points_T)
    # camera2pixel
    pixel = np.dot(in_matrix, points_T_camera).T
    pixel_xy = np.array([x / x[2] for x in pixel])[:, 0:2]
    pixel_xy = np.around(pixel_xy).astype(int)
    
    return pixel_xy

def batch_compute_box_3d(objects, ex_matrix, in_matrix):
    corners2d = []
    for obj in objects:
        corner_2d = loc_pc2img(obj, ex_matrix, in_matrix)
        corners2d.append(corner_2d)
    corners2d = np.stack(corners2d, axis=0)
    return corners2d

def draw_projected_box3d(image, qs, color=(255,255,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA) # use LINE_AA for opencv3

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.LINE_AA)
    return image