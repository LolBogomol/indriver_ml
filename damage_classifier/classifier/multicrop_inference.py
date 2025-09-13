import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image
from damage_classifier.config import DEVICE, IMG_SIZE
from damage_classifier.classifier.dataset import val_transforms


def multi_crop_predict(model, pil_img, device=DEVICE, crop_scales=[1.0, 0.7, 0.5], grid=(3,3), input_size=IMG_SIZE):
    if isinstance(pil_img, Image.Image):
        img = np.array(pil_img.convert('RGB'))
    else:
        img = pil_img.copy()
    H, W = img.shape[:2]
    crops = []
    # center crops
    for s in crop_scales:
        ch, cw = int(H * s), int(W * s)
        cy = max(0, (H-ch)//2)
        cx = max(0, (W-cw)//2)
        c = img[cy:cy+ch, cx:cx+cw]
        c = cv2.resize(c, (input_size, input_size))
        c = val_transforms(image=c)['image'].unsqueeze(0)
        crops.append(c)
    # grid patches
    gh, gw = grid
    for i in range(gh):
        for j in range(gw):
            ch = int(H / gh * 0.9)
            cw = int(W / gw * 0.9)
            y = int(i * max((H-ch)/(gh-1), 0)) if gh>1 else 0
            x = int(j * max((W-cw)/(gw-1), 0)) if gw>1 else 0
            p = img[y:y+ch, x:x+cw]
            p = cv2.resize(p, (input_size,input_size))
            p = val_transforms(image=p)['image'].unsqueeze(0)
            crops.append(p)
    batch = torch.cat(crops, dim=0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    avg_probs = probs.mean(axis=0)
    max_idx = probs[:, 2].argmax()
    top_crop = crops[max_idx][0].cpu().numpy().transpose(1, 2, 0)
    return avg_probs, probs, top_crop


def predict_image_file(model, img_path):
    pil = Image.open(img_path).convert('RGB')
    avg_probs, all_probs, top_crop = multi_crop_predict(model, pil)
    pred_class = int(np.argmax(avg_probs))
    return {"pred": pred_class, "probs": avg_probs.tolist()}
