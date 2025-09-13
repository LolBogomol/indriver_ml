import cv2
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
from damage_classifier.config import IMG_SIZE
from damage_classifier.classifier.util.set_seed import set_seed
import albumentations as A
from albumentations.pytorch import ToTensorV2


set_seed()


class MixedFullCropDataset(Dataset):
    def __init__(self, df, img_root=".", transforms_full=None, transforms_crop=None,
                 n_crops=6, crop_scales=(0.5,0.7,1.0), mode="train"):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.transforms_full = transforms_full
        self.transforms_crop = transforms_crop
        self.n_crops = n_crops
        self.scales = crop_scales
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _random_center_biased_crop(self, img, scale):
        h, w = img.shape[:2]
        ch = int(h * scale)
        cw = int(w * scale)
        # center-biased gaussian
        cy = int(np.clip(h/2 + np.random.normal(0, h*0.12), 0, h-ch))
        cx = int(np.clip(w/2 + np.random.normal(0, w*0.12), 0, w-cw))
        return img[cy:cy+ch, cx:cx+cw]

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = row['img_path']
        label = int(row['label'])
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if random.random() < 0.5:
            # full
            img_proc = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            if self.transforms_full:
                img_proc = self.transforms_full(image=img_proc)['image']
        else:
            scale = random.choice(self.scales)
            crop = self._random_center_biased_crop(img, scale)
            crop = cv2.resize(crop, (IMG_SIZE,IMG_SIZE))
            if self.transforms_crop:
                img_proc = self.transforms_crop(image=crop)['image']

        return img_proc, label


train_transforms_crop = A.Compose([
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), ratio=(0.8, 1.2), p=0.8),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=12, p=0.5),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0)),
        A.MultiplicativeNoise(multiplier=(0.9,1.1)),
    ], p=0.3),
    A.OneOf([
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
        A.GaussianBlur(blur_limit=3),
    ], p=0.3),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.6),
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
    A.Normalize(),
    ToTensorV2(),
])

train_transforms_full = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.6),
    A.Normalize(),
    ToTensorV2(),
])

val_transforms = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])
train_transforms_crop.add_targets = None
