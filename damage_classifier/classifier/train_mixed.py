import os
import torch

from torch.utils.data import DataLoader
from torch import nn
import time

from damage_classifier.services.ds_loader_service import DsLoaderService
from damage_classifier.config import DATA_DIR, SEED, N_RANDOM_CROPS, CROP_SCALES, BATCH_SIZE, DEVICE, NUM_CLASSES, \
    WEIGHT_DECAY, EPOCHS, LR, OUTPUT_DIR, BACKBONE, IMG_SIZE
from damage_classifier.classifier.dataset import MixedFullCropDataset, train_transforms_crop, train_transforms_full, \
    val_transforms
from damage_classifier.classifier.model import DamageClassifier
from damage_classifier.services.train_service import TrainService
from damage_classifier.classifier.multicrop_inference import predict_image_file


def train_mixed():
    loader_service = DsLoaderService()
    try:
        ds_cardd, df_cardd_raw = loader_service.load_cardd_hf()
        print("Loaded CarDD from HF, examples:", len(df_cardd_raw))
    except Exception as e:
        print("Failed to load CarDD via HF (offline or error):", e)

    print("Dumping HF images to disk... (this may take time and disk space)")
    df_cardd = loader_service.dump_hf_images_to_folder(df_cardd_raw, out_images_dir=os.path.join(DATA_DIR, "cardd_images"))
    print("Saved images to", os.path.join(DATA_DIR, "cardd_images"))

    df_all = df_cardd.copy()

    df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_frac = 0.12
    val_n = int(len(df_all) * val_frac)
    df_val = df_all.iloc[:val_n].reset_index(drop=True)
    df_train = df_all.iloc[val_n:].reset_index(drop=True)
    print("Train:", len(df_train), "Val:", len(df_val))

    train_dataset = MixedFullCropDataset(df_train, img_root=".", transforms_full=train_transforms_full,
                                         transforms_crop=train_transforms_crop,
                                         n_crops=N_RANDOM_CROPS, crop_scales=CROP_SCALES)
    val_dataset = MixedFullCropDataset(df_val, img_root=".", transforms_full=val_transforms,
                                       transforms_crop=val_transforms, n_crops=0, crop_scales=CROP_SCALES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = DamageClassifier().to(DEVICE)

    counts = df_train['label'].value_counts().to_dict()
    print("Class counts train:", counts)
    freqs = [counts.get(i, 1) for i in range(NUM_CLASSES)]
    inv_freq = [1.0 / f for f in freqs]
    norm = sum(inv_freq)
    class_weights = [v / norm for v in inv_freq]
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    print("Class weights:", class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = 0.0
    checkpoint_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    train = TrainService()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_f1 = train.train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_f1, val_report, val_cm = train.validate_epoch(model, val_loader, criterion, DEVICE)
        t1 = time.time()
        print(
            f"Epoch {epoch}/{EPOCHS}  train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} "
            f" val_loss={val_loss:.4f} val_f1={val_f1:.4f}  time={t1 - t0:.1f}s")
        print("Val confusion:\n", val_cm)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"model_state": model.state_dict(), "cfg": {"backbone": BACKBONE, "img_size": IMG_SIZE}},
                       checkpoint_path)
            print("Saved best model:", checkpoint_path)
        scheduler.step()

    print("Training done. Best val f1:", best_val_f1)

    print("Testing inference on few val images...")
    for i in range(min(10, len(df_val))):
        row = df_val.iloc[i]
        out = predict_image_file(model, row['img_path'])
        print("GT", row['label'], "PRED", out['pred'], "probs", out['probs'])
