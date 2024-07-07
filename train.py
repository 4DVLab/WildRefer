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
from time import time
from utils.logger import Logger
from tqdm import tqdm
from models.losses import HungarianMatcher, SetCriterion, compute_hungarian_loss
from transformers import RobertaTokenizerFast
from utils import strefer_utils

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

def compute_loss(end_points, criterion, set_criterion):
    loss, end_points = criterion(
        end_points, 6,
        set_criterion,
        query_points_obj_topk=4
    )
    return loss, end_points

def get_criterion():
    """Get loss criterion for training."""
    matcher = HungarianMatcher(1, 0, 2, True)
    losses = ['boxes', 'labels']
    losses.append('contrastive_align')
    set_criterion = SetCriterion(
        matcher=matcher,
        losses=losses, eos_coef=0.1, temperature=0.07
    )
    criterion = compute_hungarian_loss
    return criterion, set_criterion

def train_one_epoch(ep, dataloader, model, criterion, set_criterion, optimizer, scheduler, epochs, logger, verbose_step=1):
    model.train()
    for idx, input_data in enumerate(tqdm(dataloader, ncols=0, unit=' data')):
        for key in input_data:
            if isinstance(input_data[key], torch.Tensor):
                input_data[key] = input_data[key].cuda()

        optimizer.zero_grad()
        end_points = model(input_data)

        for key in input_data:
            if key not in end_points:
                end_points[key] = input_data[key]
        
        # Compute loss
        loss, end_points = compute_loss(
            end_points, criterion, set_criterion
        )

        optimizer.zero_grad()
        loss.backward()
        grad_total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 0.1
        )
        optimizer.step()
        scheduler.step()

        logger.tf_log("TrainIter/Loss", loss.item(), ep * len(dataloader) + idx)
        if idx % verbose_step == 0:
            loss_info = ''
            info = f"TRN Epoch[{ep}|{epochs}][{idx}|{len(dataloader)}] loss={round(loss.item(), 4)} "\
                   f"lr={optimizer.param_groups[0]['lr']}"
            print(' ', info)
            logger(info)

@torch.no_grad()
def evaluate(ep, model, dataset, dataloader, criterion, set_criterion, epochs, logger, best_score, name):
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
                        
        # Compute loss
        ls, _ = compute_loss(
            end_points, criterion, set_criterion
        )
        loss += ls
        total_predict_boxes.append(pred_box)
    predict_boxes = np.vstack(total_predict_boxes)
    
    acc25, acc50, m_iou = dataset.evaluate(predict_boxes)
    loss = loss / len(dataloader)

    info = f"{name} Epoch[{ep}] Acc25={acc25} Acc50={acc50} mIoU={m_iou} loss={round(loss.item(), 4)}"
    print(info)
    logger(info)
    logger.tf_log(f"{name}/Acc25", acc25, ep)
    logger.tf_log(f"{name}/Acc50", acc50, ep)
    logger.tf_log(f"{name}/mIoU", m_iou, ep)
    logger.tf_log(f"{name}/loss", loss, ep)

    if name == 'EVAL' and acc25 > best_score:
        logger.save_model(model, f"best_model.pth")
        best_score = acc25
        best_info = f"Best Epoch[{ep}] Acc25={best_score}"
        print(best_info)
        logger(best_info)

    return best_score

def main(args):
    set_random_seed(args.seed)
    print("Create Logger")
    logger = Logger(args.work_dir)
    logger(str(args))
    
    print("Create Dataset")
    train_dataset = create_dataset(args, 'train')
    generator = torch.Generator()
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, generator=generator)

    print("Create Model")
    model = create_model(args)
    if args.pretrain:
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.pretrain, map_location='cpu')['model'], strict=False)
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")
    
    param_dicts = [
            {"params": [
                    p for n, p in model.named_parameters() if "point_backbone_net" not in n and "text_encoder" not in n and p.requires_grad
                ]
            },
            {"params": [
                    p for n, p in model.named_parameters() if "point_backbone_net" in n and p.requires_grad
                ],
                "lr": args.lr_backbone
            },
            {"params": [
                    p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad
                ],
                "lr": args.text_encoder_lr
            }
        ]
    print("Create Optimizer and Scheduler")
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            gamma=0.1,
            milestones=[(m - args.warmup_epoch) * len(train_loader) for m in args.lr_step])
    criterion, set_criterion = get_criterion()

    best_score = -1
    start_epoch = 0
    model.cuda()
    print("Start to train the model")
    for i in range(start_epoch, args.epochs):
        ep = i + 1
        train_one_epoch(ep, train_loader, model, criterion, set_criterion, optimizer, scheduler, args.epochs, logger, args.verbose_step)
        if ep % 1 == 0:
            logger.save_model(model, f"epoch_{ep}_model.pth", epoch=ep, best_score=best_score,\
                                criterion=criterion, optimizer=optimizer, scheduler=scheduler)
    return

if __name__ == '__main__':
    args = get_args_parser()
    main(args)