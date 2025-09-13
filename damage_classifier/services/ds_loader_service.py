from datasets import load_dataset
import torch
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from PIL import Image
import io


class DsLoaderService:
    @staticmethod
    def load_cardd_hf():
        """
        Loads the CarDD dataset from Hugging Face if available.
        Returns a pandas DataFrame with columns: ['img_path','label'] where label is 0/1/2 (we'll map).
        For CarDD we map instances: scratch/dent -> damage; CarDD has instance annotations, we'll create
        image-level labels: 0 if no damage, 1 for minor (we heuristically map small boxes), 2 for severe (large boxes).
        """
        ds = load_dataset("harpreetsahota/CarDD")

        rows = []
        for split in ds.keys():
            print("Processing split:", split)
            for ex in tqdm(ds[split]):
                img_id = ex['image']['id'] if isinstance(ex['image'], dict) else None
                img = ex['image']

                anns = ex.get('annotations', [])
                if not anns:
                    label = 0
                else:
                    h = ex['image']['height']
                    w = ex['image']['width']
                    max_area = 0
                    total_area = 0
                    for a in anns:
                        x, y, ww, hh = a['bbox']
                        area = ww * hh
                        total_area += area
                        if area > max_area: max_area = area
                    frac = max_area / (w * h)
                    if frac > 0.02:
                        if frac > 0.06:
                            label = 2   # severe
                        else:
                            label = 1   # minor
                    else:
                        if total_area / (w*h) > 0.01:
                            label = 1
                        else:
                            label = 1
                rows.append({"hf_example": ex, "label": label})
        df = pd.DataFrame(rows)
        return ds, df

    @staticmethod
    def dump_hf_images_to_folder(df_hf, out_images_dir="data/cardd_images"):
        out_images_dir = Path(out_images_dir)
        out_images_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, r in tqdm(df_hf.iterrows(), total=len(df_hf)):
            ex = r['hf_example']
            label = int(r['label'])
            img_obj = ex['image']
            try:
                if isinstance(img_obj, dict) and 'bytes' in img_obj:
                    im = Image.open(io.BytesIO(img_obj['bytes'])).convert('RGB')
                else:
                    im = Image.fromarray(img_obj['array']) if isinstance(img_obj, dict) and 'array' in img_obj else None
            except Exception:
                im = None
            if im is None:
                try:
                    im = Image.open(ex['image']['path']).convert('RGB')
                except Exception:
                    try:
                        im = ex['image']
                    except Exception:
                        raise RuntimeError("Cannot extract image bytes for example idx=%s" % idx)
            fname = out_images_dir / f"{idx:06d}.jpg"
            im.save(fname, quality=95)
            rows.append({"img_path": str(fname), "label": label})

        return pd.DataFrame(rows)
