import argparse
import os
import shutil
import random


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description='Splitting dataset')
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument("--train", type=str, default='data/npy/train/reflectivity')
    parser.add_argument("--val", type=str, default='data/npy/val/reflectivity')
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--nproc", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # random seed
    random.seed(args.seed)

    # source destination
    train_dir = args.train
    val_dir = args.val
    os.makedirs(val_dir, exist_ok=True)

    # get data
    all_images = os.listdir(train_dir)
    all_images = sorted(all_images, key=lambda x: int(x.split('.')[0]))

    # split ratio
    val_ratio = args.ratio

    # random split
    val_size = int(len(all_images) * val_ratio)
    val_images = random.sample(all_images, val_size)

    # move
    for image in val_images:
        src_path = os.path.join(train_dir, image)
        dst_path = os.path.join(val_dir, image)
        shutil.move(src_path, dst_path)
        print(f'Move {src_path} data to {dst_path}')


if __name__ == '__main__':
    main()
