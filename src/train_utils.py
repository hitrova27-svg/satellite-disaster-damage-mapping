"""Shared training loop + metrics, used by notebooks 03 and 04."""
from pathlib import Path
import torch
import torch.nn as nn


def train_model(model, train_dl, val_dl, device, epochs=20, lr=1e-3,
                class_weights=None, ckpt_path=None):
    """Generic train loop for the (pre, post, label) siamese models."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    w = class_weights.to(device) if class_weights is not None else None
    crit = nn.CrossEntropyLoss(weight=w)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best = 0.0
    for ep in range(epochs):
        model.train(); tl = 0.0
        for pre, post, y in train_dl:
            pre, post, y = pre.to(device), post.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(pre, post), y)
            loss.backward(); opt.step()
            tl += loss.item() * len(y)
        vl, vacc = _validate(model, val_dl, device, crit)
        history["train_loss"].append(tl / len(train_dl.dataset))
        history["val_loss"].append(vl); history["val_acc"].append(vacc)
        print(f"epoch {ep+1:02d}  train_loss {history['train_loss'][-1]:.3f}"
              f"  val_loss {vl:.3f}  val_acc {vacc:.3f}")
        if ckpt_path and vacc > best:
            best = vacc
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
    return history


@torch.no_grad()
def _validate(model, dl, device, crit):
    model.eval(); loss = 0.0; correct = 0; n = 0
    for pre, post, y in dl:
        pre, post, y = pre.to(device), post.to(device), y.to(device)
        out = model(pre, post)
        loss += crit(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def evaluate(model, dl, device, class_names=None):
    """Print accuracy, per-class precision/recall/F1 and a confusion matrix."""
    from sklearn.metrics import classification_report, confusion_matrix
    model.eval(); ys = []; ps = []
    for pre, post, y in dl:
        out = model(pre.to(device), post.to(device))
        ys.extend(y.tolist()); ps.extend(out.argmax(1).cpu().tolist())
    print(classification_report(ys, ps, target_names=class_names, zero_division=0))
    print("confusion matrix:\n", confusion_matrix(ys, ps))
    return ys, ps


def plot_history(history):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(history["val_acc"]); ax[1].set_title("val accuracy")
    plt.tight_layout(); plt.show()
