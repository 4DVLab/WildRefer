import os
from torch.utils.data import Dataset
import json
import torch
import numpy as np
import cv2
import pickle
import json
from utils import strefer_utils, pc_utils
from tqdm import tqdm
from transformers import RobertaTokenizerFast
import spacy
from utils.box_util import resize_img_keep_ratio

cv2.ocl.setUseOpenCL(False)   
cv2.setNumThreads(0)


SRC_PATH = "src/STRefer"

class STReferDataset(Dataset):
    def __init__(self, args, split="train") -> None:
        super().__init__()
        self.args = args
        if split == "train":
            self.dataset = json.load(open("data/strefer_train.json"))
        else:
            self.dataset = json.load(open("data/strefer_test.json"))

        self.find_previous = json.load(open(os.path.join(SRC_PATH, "find_previous_strefer.json")))
        self.points2image = json.load(open(os.path.join(SRC_PATH, "points2image_strefer.json")))
        self.max_objects = args.max_obj_num
        self.max_lang_num = args.max_lang_num
        self.frame_num = args.frame_num

        self.range = [16.36, 0, -1.5, 30.72, 40.96, 5, 0]

        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.nlp = spacy.load('en_core_web_sm')
        
    
    def __getitem__(self, index):
        data = self.dataset[index]
        data_dict = {}

        scene_id = data['scene_id']
        point_cloud_name = data['point_cloud']['point_cloud_name']
        image_name = data['image']['image_name']
        bbox = data['point_cloud']['bbox']
        object_id = data['object_id']
        description = data['language']['description'].lower()
        ann_id = str(data['language']['ann_id'])

        target_bbox = np.array(bbox, dtype=np.float32)

        # point cloud
        scene_file = os.path.join(SRC_PATH, 'points_rgbd', scene_id, f"{point_cloud_name}.npy")
        scene = np.load(scene_file)
        scene[:, 3:6] = scene[:, 3:6] / 255.
        scene = strefer_utils.random_sampling(scene, 30000)

        # images
        image_path = os.path.join(SRC_PATH, 'image', scene_id, f'{image_name}.jpg')   
        image = strefer_utils.load_image(image_path)
        image, ratio, pad_w, pad_h = resize_img_keep_ratio(image, self.args.img_size)
        image = np.transpose(image, (2, 0, 1))
        img_mask = np.zeros(image.shape[1:3], dtype=bool)
        img_mask[0+pad_h//2:image.shape[0]-pad_h//2, 0+pad_w//2:image.shape[1]-pad_w//2] = 1

        scenes = [scene]
        images = [image]
        images_mask = [img_mask]
        dynamic_mask = [1]
        for _ in range(1, self.frame_num):
            if point_cloud_name:
                point_cloud_name = self.find_previous[scene_id][point_cloud_name]
                image_name = self.points2image[scene_id][point_cloud_name]
            if point_cloud_name:
                scene_file = os.path.join(SRC_PATH, 'points_rgbd', scene_id, f"{point_cloud_name}.npy")
                add_scene = np.load(scene_file)
                add_scene[:, 3:6] = add_scene[:, 3:6] / 255.
                add_scene = strefer_utils.random_sampling(scene, 30000)
                dynamic_mask.append(1)

                image_path = os.path.join(SRC_PATH, 'image', scene_id, f'{image_name}.jpg')
                image = strefer_utils.load_image(image_path)
                image, ratio, pad_w, pad_h = resize_img_keep_ratio(image, self.args.img_size)
                image = np.transpose(image, (2, 0, 1))
                img_mask = np.zeros(image.shape[1:3], dtype=bool)
                img_mask[0+pad_h//2:image.shape[0]-pad_h//2, 0+pad_w//2:image.shape[1]-pad_w//2] = 1
            else:
                add_scene = np.zeros((30000, 6), dtype=np.float32)
                image = np.zeros((3, self.args.img_size, self.args.img_size), dtype=np.float32)
                img_mask = np.zeros((self.args.img_size, self.args.img_size), dtype=bool)
                img_mask[0, 0] = True
                dynamic_mask.append(0)
            scenes.append(add_scene)
            images.append(image)
            images_mask.append(img_mask)
        scenes = np.stack(scenes, axis=0)
        dynamic_mask = np.hstack(dynamic_mask)
        images = np.stack(images, axis=0)
        images_mask = np.stack(images_mask, axis=0)

        # language
        text = ' '.join(description.replace(',', ' ,').replace('.', ' .').split()) + ' not mentioned'

        data_dict['point_clouds'] = scenes.astype(np.float32)
        data_dict['text'] = text
        data_dict['dynamic_mask'] = dynamic_mask.astype(np.int64)
        data_dict['image'] = images.astype(np.float32)
        data_dict['img_mask'] = images_mask

        # GT
        gt_boxes3d = np.zeros((self.max_objects, 6))
        gt_boxes3d[0] = target_bbox[:6]
        bbox_label_mask = np.zeros((self.max_objects, ))
        bbox_label_mask[0] = 1
        point_instance_label = -np.ones(len(scene))
        _, instance_ind = strefer_utils.extract_pc_in_box3d(
            scene.copy(), strefer_utils.my_compute_box_3d(target_bbox[0:3], target_bbox[3:6], target_bbox[6])
            )
        point_instance_label[instance_ind] = 0

        data_dict['center_label'] = gt_boxes3d[:, :3].astype(np.float32)
        data_dict['size_gts'] = gt_boxes3d[:, 3:6].astype(np.float32)
        data_dict['box_label_mask'] = bbox_label_mask.astype(np.float32)
        data_dict['point_instance_label'] = point_instance_label.astype(np.int64)

        _labels = np.zeros(self.max_objects)
        data_dict['sem_cls_label'] = _labels.astype(np.int64)
        tokens_positive, positive_map = self._get_token_positive_map(description, self.max_lang_num)
        data_dict['tokens_positive'] = tokens_positive.astype(np.int64)
        data_dict['positive_map'] = positive_map.astype(np.float32)
        
        return data_dict
    
    def _get_token_positive_map(self, description, max_lang_num):

        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = ' '.join(description.replace(',', ' ,').split())
        caption = ' ' + caption + ' '
        tokens_positive = np.zeros((self.max_objects, 2))

        doc = self.nlp(caption)
        cat_names = []
        for token in doc:
            if token.dep_ == 'nsubj':
                cat_names.append(token.text)
                break
        if len(cat_names) <= 0:
            for token in doc:
                if token.dep_ == 'ROOT':
                    cat_names.append(token.text)
                    break
        
        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(' ' + cat_name + ' ')
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(' ' + cat_name)
                len_ = len(caption[start_span+1:].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != ' ':
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != ' ':
                    len_ += 1
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span
        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(description.replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )
        positive_map = np.zeros((self.max_objects, max_lang_num))
        gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)], max_lang_num)
        positive_map[:len(cat_names)] = gt_map

        return tokens_positive, positive_map
    
    def evaluate(self, predict_boxes, output_path=''):
        pred_boxes = []
        target_boxes = []
        eval_results = []
        idx = 0
        for pred_box in tqdm(predict_boxes):
            data = self.dataset[idx]
            scene_id = data['scene_id']
            point_cloud_info = data['point_cloud']
            image_info = data['image']
            point_cloud_name = point_cloud_info['point_cloud_name']
            image_name = image_info['image_name']

            target = point_cloud_info['bbox']

            if output_path:
                ex_matrix = data['calibration']['ex_matrix']
                in_matrix = data['calibration']['in_matrix']
                language = data['language']['description']
                point_cloud_path = os.path.join(SRC_PATH, 'points_rgbd', scene_id, f"{point_cloud_name}.npy")
                img_name = os.path.join(SRC_PATH, 'image', scene_id, f"{image_name}.jpg")
                out_data = dict()
                out_data['gt_box'] = target
                out_data['pred_box'] = pred_box
                out_data["point_cloud_path"] = point_cloud_path
                out_data["image_path"] = img_name
                out_data['gt_corner2d'] = strefer_utils.batch_compute_box_3d([np.array(target)], ex_matrix, in_matrix)
                out_data['pred_corner2d'] = strefer_utils.batch_compute_box_3d([np.array(pred_box)], ex_matrix, in_matrix)
                out_data['language'] = language
                out_data['iou'] = pc_utils.cal_iou3d(pred_box, target)
                eval_results.append(out_data)

            target_boxes.append(target)
            idx += 1

        target_boxes = np.array(target_boxes)
        pred_boxes = predict_boxes

        if output_path:
            save_pkl(eval_results, output_path)
        acc25, acc50, miou = pc_utils.cal_accuracy(pred_boxes, target_boxes)
        return acc25, acc50, miou
        
    def __len__(self):
        return len(self.dataset)

def save_pkl(file, output_path):
    output = open(output_path, 'wb')
    pickle.dump(file, output)
    output.close()

def get_positive_map(tokenized, tokens_positive, max_lang_num):
    """Construct a map of box-token associations."""
    positive_map = torch.zeros((len(tokens_positive), max_lang_num), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()