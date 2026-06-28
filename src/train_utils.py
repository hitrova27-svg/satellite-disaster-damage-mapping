"""Shared training loop + metrics (numpy only, no sklearn needed)."""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn


def train_model(model, train_dl, val_dl, device, epochs=5, lr=1e-3,
                class_weights=None, ckpt_path=None, verbose=True):
    """Generic train loop for the (pre, post, label) siamese models."""
    model = model.to(device)
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
        if verbose:
            print(f"  epoch {ep+1:02d}/{epochs}  train_loss {history['train_loss'][-1]:.3f}"
                  f"  val_loss {vl:.3f}  val_acc {vacc:.2f}")
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
def predict(model, dl, device):
    """Run the model over a loader -> (y_true, y_pred) numpy arrays."""
    model.eval(); ys, ps = [], []
    for pre, post, y in dl:
        out = model(pre.to(device), post.to(device))
        ys += y.tolist(); ps += out.argmax(1).cpu().tolist()
    return np.array(ys), np.array(ps)


def scores(y_true, y_pred, n_classes=4):
    """Return (confusion_matrix, accuracy, macro_f1, per_class_f1)."""
    cm = np.zeros((n_classes, n_classes), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = np.trace(cm) / max(cm.sum(), 1)
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        prec = tp / max(cm[:, c].sum(), 1)
        rec = tp / max(cm[c, :].sum(), 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
    return cm, acc, float(np.mean(f1s)), f1s


def plot_history(history):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(history["train_loss"], label="train"); ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("loss"); ax[0].legend()
    ax[1].plot(history["val_acc"]); ax[1].set_title("val accuracy")
    plt.tight_layout(); plt.show()


def plot_confusion(cm, class_names, ax=None, title=""):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return ax
