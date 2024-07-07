import os
import sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import random
import torch
from datasets import create_dataset
from models import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_args_parser():
    parser = argparse.ArgumentParser('Set config')
    parser.add_argument('--dataset', default='', type=str)
    parser.add_argument('--img_size', default=384, type=int)
    parser.add_argument('--max_obj_num', default=100, type=int)
    parser.add_argument('--max_lang_num', default=100, type=int)
    parser.add_argument('--num_queries', default=256, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--frame_num', default=3, type=int)
    parser.add_argument('--dynamic', default=True, action='store_true')

    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-3, type=float)
    parser.add_argument('--text_encoder_lr', default=1e-5, type=float)
    parser.add_argument('--lr_step', default=[45, 80], type=int, nargs='+')
    parser.add_argument('--warmup-epoch', type=int, default=-1)


    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--val_epoch', default=1, type=int)
    parser.add_argument('--verbose_step', default=10, type=int)
    parser.add_argument('--pretrain', default='', type=str)
    parser.add_argument('--work_dir', default='outputs/debug', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--butd', action='store_true')
    args = parser.parse_args()
    if args.debug:
        args.work_dir = "debug"
        args.num_workers = 0
        args.batch_size = 2
    return args


@torch.no_grad()
def evaluate(args, model, dataset, dataloader):
    model.eval()
    loss = 0
    total_predict_boxes = []
    for input_data in tqdm(dataloader, colour='red', unit=' data'):
        for key in input_data:
            if isinstance(input_data[key], torch.Tensor):
                input_data[key] = input_data[key].cuda()

        end_points = model(input_data)

        for key in input_data:
            if key not in end_points:
                end_points[key] = input_data[key]
        
        # contrast
        pred_center = end_points['last_center'].detach().cpu()
        pred_size = end_points["last_pred_size"].detach().cpu()
        pred_boxes = torch.concat([pred_center, pred_size], dim=-1).numpy()

        proj_tokens = end_points['proj_tokens']  # (B, tokens, 64)
        proj_queries = end_points['last_proj_queries']  # (B, Q, 64)
        sem_scores = torch.matmul(proj_queries, proj_tokens.transpose(-1, -2))
        sem_scores_ = sem_scores / 0.07  # (B, Q, tokens)
        sem_scores = torch.softmax(sem_scores_, dim=-1)

        token = end_points['tokenized']
        mask = token['attention_mask'].detach().cpu()
        last_pos = mask.sum(1) - 2

        bs = sem_scores.shape[0]
        pred_box = np.zeros((bs, 7))
        for i in range(bs):
            sim = 1 - sem_scores[i, :, last_pos[i]]
            max_idx = torch.argmax(sim)
            box = pred_boxes[i, max_idx.item()]
            pred_box[i, :6] = box

        total_predict_boxes.append(pred_box)
    predict_boxes = np.vstack(total_predict_boxes)
    acc25, acc50, m_iou = dataset.evaluate(predict_boxes)  
    loss = loss / len(dataloader)

    info = f"Acc25={acc25} Acc50={acc50} mIoU={m_iou}"
    print(info)

def main(args):
    set_random_seed(args.seed)

    print("Create Dataset")
    test_dataset = create_dataset(args, 'test')
    generator = torch.Generator()
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, generator=generator)

    print("Create Model")
    model = create_model(args)
    model.load_state_dict(torch.load(args.pretrain, map_location='cpu')['model'], strict=True)

    model.cuda()
    evaluate(args, model, test_dataset, test_loader)

    return

if __name__ == '__main__':
    args = get_args_parser()
    main(args)