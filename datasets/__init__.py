from .liferefer_dataset import LifeReferDataset
from .strefer_dataset import STReferDataset

def create_dataset(args, split):
    if args.dataset == 'liferefer':
        return LifeReferDataset(args, split)
    elif args.dataset == 'strefer':
        return STReferDataset(args, split)
    else:
        raise ValueError("Wrong Dataset")