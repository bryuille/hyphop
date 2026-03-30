import os
import shutil
import urllib.request
import zipfile
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def ensure_tiny_imagenet(root: str) -> str:
    """
    Ensures Tiny ImageNet is present under:
      <root>/tiny-imagenet-200/

    Downloads + extracts if needed and returns the extracted directory path.
    """
    dest = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(dest):
        return dest

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")

    if not os.path.isfile(zip_path):
        print(f"Downloading Tiny ImageNet (~236 MB) to {zip_path} ...")
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    print("Extraction complete.")
    return dest


def ensure_val_reorg(tiny_root: str) -> str:
    """
    Tiny ImageNet ships val images in a flat directory. This creates:
      <tiny_root>/val/images_by_class/<class_id>/*.JPEG

    Returns the reorg directory path.
    """
    by_class = os.path.join(tiny_root, "val", "images_by_class")
    if os.path.isdir(by_class):
        return by_class

    ann_file = os.path.join(tiny_root, "val", "val_annotations.txt")
    img_dir = os.path.join(tiny_root, "val", "images")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"Missing val annotations at {ann_file}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing val images directory at {img_dir}")

    print("Reorganising Tiny ImageNet val set (one-time) ...")
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            fname, cls = parts[0], parts[1]
            cls_dir = os.path.join(by_class, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(img_dir, fname)
            dst = os.path.join(cls_dir, fname)
            if not os.path.isfile(dst):
                shutil.copy(src, dst)
    print("Val reorganisation complete.")
    return by_class


def get_tiny_imagenet_loaders(
    data_dir: str,
    batch_size: int = 64,
    test_batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, int, int]:
    tiny_root = ensure_tiny_imagenet(data_dir)
    val_dir = ensure_val_reorg(tiny_root)

    tf_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_ds = datasets.ImageFolder(os.path.join(tiny_root, "train"), transform=tf_train)
    test_ds = datasets.ImageFolder(val_dir, transform=tf_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    in_dim = 64 * 64 * 3
    num_classes = 200
    return train_loader, test_loader, in_dim, num_classes

