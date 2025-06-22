#!/usr/bin/env python3
"""Train, hyper-parameter search, and fine-tune a ResNet-18 model for binary
chest-X-ray classification (NORMAL vs PNEUMONIA) with multi-GPU support.

Usage (single GPU / CPU):
    python train_resnet_chestxray.py

For multi-GPU DataParallel training, the same command works automatically if
multiple CUDA devices are visible.

The script expects the following directory layout *after* running the AWS S3
sync commands shown below (uncomment and run once):

    data/
      ├── train/
      │     ├── NORMAL/    <image files>
      │     └── PNEUMONIA/ <image files>
      └── test/
            ├── NORMAL/    <image files>
            └── PNEUMONIA/ <image files>

If your images are all in a flat folder, use the commented block in `main()`
to reorganise them automatically.
"""
from __future__ import annotations

import gc
import json
import pathlib
import random
import shutil
from typing import Dict, Tuple, List

import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets

# ──────────────────────────────────────────────────────────────────────────────
# Globals & reproducibility
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# PIL — tolerate truncated / partially-downloaded images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Normalisation constants from ImageNet
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# Data transforms
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def validate_model(model: nn.Module, loader: DataLoader, criterion) -> Tuple[float, float]:
    """Return (accuracy, avg_loss) on *loader*."""
    model.eval()
    correct = total = 0
    running_loss: float = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total, running_loss / len(loader)

# ──────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────────────────────────

def objective(trial: optuna.trial.Trial) -> float:
    # Hyper-parameters
    lr           = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [16, 32, 64])
    optim_name   = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    m1 = trial.suggest_int("milestone1", 15, 35)
    m2 = trial.suggest_int("milestone2", m1 + 5, 45)
    gamma = trial.suggest_float("gamma", 0.05, 0.3)

    patience          = trial.suggest_int("patience", 5, 15)
    validation_freq   = trial.suggest_int("validation_freq", 3, 8)

    print(f"\n── Trial {trial.number} ──")
    print(json.dumps({
        "lr": lr, "batch_size": batch_size, "optimizer": optim_name,
        "weight_decay": weight_decay, "milestones": [m1, m2],
        "gamma": gamma, "patience": patience, "val_freq": validation_freq
    }, indent=2))

    # Datasets / loaders
    train_loader = DataLoader(
        datasets.ImageFolder("data/train", transform=train_tfms),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.ImageFolder("data/test", transform=val_tfms),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Model
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    if DEVICE.startswith("cuda") and torch.cuda.device_count() > 1:
        print(f"→ Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model.to(DEVICE)

    # Optimiser, scheduler, loss
    if optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2], gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    best_acc, patience_ctr = 0.0, 0

    try:
        for epoch in range(1, 21):  # max epochs for HPO
            # Training pass
            model.train()
            run_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                run_loss += loss.item()
            scheduler.step()

            # Validation
            if epoch % validation_freq == 0:
                val_acc, val_loss = validate_model(model, val_loader, criterion)
                train_loss = run_loss / len(train_loader)
                print(f"Epoch {epoch:2d} | train={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

                improved = val_acc > best_acc
                best_acc = max(best_acc, val_acc)
                patience_ctr = 0 if improved else patience_ctr + 1
                print("  ↳", "improved" if improved else f"no-improve ({patience_ctr}/{patience})")

                trial.report(val_acc, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if patience_ctr >= patience:
                    print("  ↳ early-stop HPO")
                    break

        # final check if last epoch skipped validation
        if epoch % validation_freq != 0:
            last_acc, _ = validate_model(model, val_loader, criterion)
            best_acc = max(best_acc, last_acc)

        return best_acc

    except optuna.exceptions.TrialPruned:
        raise

    except Exception as exc:
        print(f"Trial {trial.number} failed: {exc}")
        return 0.0

    finally:
        # memory cleanup
        del model, train_loader, val_loader, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ──────────────────────────────────────────────────────────────────────────────
# Full training after HPO
# ──────────────────────────────────────────────────────────────────────────────

def train_full(best_params: Dict[str, object]):
    """Retrain on full data with the best hyper-parameters."""
    print("\nRetraining with best parameters:")
    print(json.dumps(best_params, indent=2))

    # Hyper-parameters
    lr             = best_params["lr"]
    batch_size     = best_params["batch_size"]
    optim_name     = best_params["optimizer"]
    weight_decay   = best_params["weight_decay"]
    m1             = best_params.get("milestone1", 30)
    m2             = best_params.get("milestone2", 40)
    gamma          = best_params.get("gamma", 0.1)
    patience_epochs = best_params.get("patience", 5) * best_params.get("validation_freq", 5)

    # Datasets / loaders
    train_loader = DataLoader(
        datasets.ImageFolder("data/train", transform=train_tfms),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.ImageFolder("data/test", transform=val_tfms),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Model
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, 2)
    if DEVICE.startswith("cuda") and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    model.to(DEVICE)

    # Optimiser / scheduler
    if optim_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m1, m2], gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    best_acc, patience_ctr = 0.0, 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accs:   List[float] = []

    print("\nStarting full training (max 100 epochs)…\n" + "-" * 60)
    for epoch in range(1, 101):
        # Train
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        scheduler.step()

        # Validate
        val_acc, val_loss = validate_model(model, val_loader, criterion)
        train_loss = run_loss / len(train_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | lr={current_lr:.6f}")

        # Checkpoint / early stop
        if val_acc > best_acc:
            best_acc = val_acc
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_acc": best_acc,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_accs": val_accs,
                "best_params": best_params,
            }, "best_model.pth")
            print(f"          → New best model saved ({best_acc:.4f})")
        else:
            patience_ctr += 1

        if patience_ctr >= patience_epochs:
            print(f"\nEarly-stopping after {patience_ctr} no-improve epochs → best_acc={best_acc:.4f}")
            break

    print("-" * 60)
    print(f"Training finished • best_val_acc={best_acc:.4f}")

    # Final eval using best ckpt
    ckpt = torch.load("best_model.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    final_acc, final_loss = validate_model(model, val_loader, criterion)
    print(f"Final accuracy={final_acc:.4f} | final loss={final_loss:.4f}")

    # Training curves
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, val_accs, label="Val Acc")
    ax2.axhline(best_acc, ls="--", color="red", label=f"Best={best_acc:.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved as training_curves.png")

# ──────────────────────────────────────────────────────────────────────────────
# Main entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # OPTIONAL: one-time dataset re-organisation from flat → class folders
    # base = pathlib.Path("data/train")
    # for f in base.iterdir():
    #     if f.is_file():
    #         label = "NORMAL" if "normal" in f.name.lower() else "PNEUMONIA"
    #         dest = base / label
    #         dest.mkdir(exist_ok=True)
    #         shutil.move(str(f), dest / f.name)

    # AWS S3 sync example (uncomment to use)
    # !aws s3 sync s3://victor-medical-ai-chest-xray/train/ ./data/train/ --no-progress --only-show-errors
    # !aws s3 sync s3://victor-medical-ai-chest-xray/test/  ./data/test/  --no-progress --only-show-errors

    # Hyper-parameter optimisation
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print("\n" + "=" * 60)
    print(f"HPO finished » best_trial={study.best_trial.number} | best_acc={study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Full retraining
    train_full(study.best_params)

if __name__ == "__main__":
    main()
