from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report
import numpy as np


class TrainService:
    @staticmethod
    def train_epoch(model, loader, optimizer, criterion, device):
        model.train()
        losses = []
        all_preds = []
        all_targets = []
        pbar = tqdm(loader)
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(yb.detach().cpu().numpy().tolist())
            pbar.set_description(f"loss {np.mean(losses):.4f}")
        f1 = f1_score(all_targets, all_preds, average='macro')
        return np.mean(losses), f1

    @staticmethod
    def validate_epoch(model, loader, criterion, device):
        model.eval()
        losses = []
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in tqdm(loader):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                losses.append(loss.item())
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_targets.extend(yb.cpu().numpy().tolist())
        all_probs = np.concatenate(all_probs, axis=0)
        preds = all_probs.argmax(axis=1)
        f1 = f1_score(all_targets, preds, average='macro')
        report = classification_report(all_targets, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_targets, preds)
        return np.mean(losses), f1, report, cm
