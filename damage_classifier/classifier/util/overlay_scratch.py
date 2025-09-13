import random
import cv2


def overlay_scratch(img, severity='minor'):
    """
    img: numpy RGB uint8
    severity: 'minor' or 'severe'
    returns augmented image (numpy)
    Draws synthetic scratches or dents on the image.
    """
    out = img.copy()
    h, w = out.shape[:2]
    if severity == 'minor':
        # draw 1-3 thin strokes
        n = random.randint(1, 3)
        for _ in range(n):
            x1 = random.randint(int(w*0.1), int(w*0.9))
            y1 = random.randint(int(h*0.2), int(h*0.8))
            x2 = min(w-1, x1 + random.randint(30, int(w*0.4)))
            y2 = min(h-1, y1 + random.randint(-20,20))
            thickness = random.uniform(1.0, 3.0)
            color = (255, 255, 255) if random.random() < 0.6 else (200,200,200)
            cv2.line(out, (x1, y1), (x2, y2), color, int(thickness), lineType=cv2.LINE_AA)
            # slight blur to simulate paint scuff
            cv2.GaussianBlur(out, (3, 3), 0, dst=out)
    else:
        # severe: larger jagged stroke or ellipse (deep scratch/dent)
        center = (random.randint(int(w*0.2), int(w*0.8)), random.randint(int(h*0.2), int(h*0.8)))
        axes = (random.randint(int(w*0.05), int(w*0.2)), random.randint(int(h*0.03), int(h*0.12)))
        angle = random.randint(0,180)
        color = (20,20,20)
        cv2.ellipse(out, center, axes, angle, 0, 360, color, -1)
        cv2.GaussianBlur(out, (5,5), 0, dst=out)
    return out


class SyntheticScratch:
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, image, **kwargs):
        if random.random() < self.p:
            sev = 'minor' if random.random() < 0.7 else 'severe'
            image = overlay_scratch((image * 255).astype('uint8'), severity=sev)
            image = image.astype('float32') / 255.0
        return {"image": image}
