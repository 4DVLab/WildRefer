import os
import os.path as osp
import datetime
from tensorboardX import SummaryWriter
import torch

class Logger:
    def __init__(self, work_dir=None) -> None:
        current_date = datetime.datetime.now()
        month = current_date.month
        day = current_date.day
        hour = current_date.hour
        minute = current_date.minute
        sec = current_date.second
        self.work_dir = work_dir
        if self.work_dir is None:
            self.work_dir = f"work_dir/{month}-{day}-{hour}-{minute}-{sec}"
        else:
            self.work_dir = osp.join(self.work_dir, f"{month}-{day}-{hour}-{minute}-{sec}")
        if not osp.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.log = osp.join(self.work_dir, "log.log")

        self.tensorboard_log = SummaryWriter(self.work_dir)

        f = open(self.log, 'w')
        f.close()
    
    def __call__(self, info):
        with open(self.log, 'a') as f:
            info += "\n"
            f.write(info)
    
    def tf_log(self, key, value, iter):
        self.tensorboard_log.add_scalar(key, value, iter)

    
    def save_model(self, model, path, epoch=None, best_score=None,\
                            criterion=None, optimizer=None, scheduler=None):
        state = dict()
        state['model'] = model.state_dict()
        if epoch is not None:
            state['epoch'] = epoch
        if best_score is not None:
            state['best_score'] = epoch
        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()
        torch.save(state, osp.join(self.work_dir, path))
    
    def load_checkpoint(self, model, path, criterion=None, optimizer=None, scheduler=None):
        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state['model'])
        epoch = state['epoch']
        best_score = state['best_score']
        criterion.load_state_dict(state['criterion'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        return epoch, best_score
    
    def load_model(self, model, path):
        return model.load_state_dict(torch.load(path)['model'])