"""CT scan classifier: TinyVGG baseline (nCT / NiCT / pCT)."""
#--------------------------------------
#Imports
#--------------------------------------
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchinfo import summary
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
HIDDEN_UNITS = 10
USE_CLASS_WEIGHTS = False  
SEED = 42
SAVE_PATH = "tinyvgg_best.pth"
NUM_WORKERS = 6

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=output_shape),  # in_features inferred at first forward pass
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------------
# Train / eval steps
# ----------------------------------------------------------------------
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0.0
    all_preds, all_labels = [], []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_preds.append(y_pred.argmax(dim=1).cpu())
        all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    train_loss /= len(dataloader)
    train_acc = (all_preds == all_labels).float().mean().item()
    train_f1 = f1_score(all_labels, all_preds, average="macro")
    return train_loss, train_acc, train_f1


def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0.0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            test_loss += loss_fn(logits, y).item()

            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_loss /= len(dataloader)
    test_acc = (all_preds == all_labels).float().mean().item()
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    return test_loss, test_acc, test_f1, all_labels, all_preds


def train(model, train_dataloader, val_dataloader, optimizer, loss_fn,
          epochs, device, save_path):
    results = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
    }
    best_f1 = 0.0

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_f1 = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        val_loss, val_acc, val_f1, _, _ = test_step(
            model, val_dataloader, loss_fn, device
        )

        print(
            f"Epoch {epoch} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )

        for key, value in zip(
            results.keys(),
            [train_loss, train_acc, train_f1, val_loss, val_acc, val_f1],
        ):
            results[key].append(value)

        # Save the best model according to validation macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved (val F1 = {best_f1:.4f})")

    return results


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_results(results):
    epochs = range(len(results["train_loss"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, metric in zip(axes, ["loss", "acc", "f1"]):
        ax.plot(epochs, results[f"train_{metric}"], label="train")
        ax.plot(epochs, results[f"val_{metric}"], label="val")
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    torch.backends.cudnn.benchmark = True
    print(f"Available device is {device}")
    torch.manual_seed(SEED)

    # Dataset paths
    data_path = Path(__file__).parent / "data"
    images_path = data_path / "Original CT Scans"
    print(images_path, images_path.exists())

    # Augmentation only for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    # No augmentation for validation/test
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Two ImageFolder instances: same files and class order, different transforms
    train_dataset = datasets.ImageFolder(root=images_path, transform=train_transform)
    eval_dataset = datasets.ImageFolder(root=images_path, transform=eval_transform)
    class_names = train_dataset.classes
    print(train_dataset.class_to_idx)

    # Stratified split on indices (train 70% / val 15% / test 15%)
    indices = list(range(len(train_dataset)))
    labels = train_dataset.targets

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=[labels[i] for i in temp_idx], random_state=SEED
    )

    # Same indices, different transforms
    train_ds = Subset(train_dataset, train_idx)
    val_ds = Subset(eval_dataset, val_idx)
    test_ds = Subset(eval_dataset, test_idx)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True )

    print(f"Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} images, {len(val_loader)} batches")
    print(f"Test:  {len(test_ds)} images, {len(test_loader)} batches")

    # Model
    model_0 = TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names),
    ).to(device)

    # Dummy forward pass (initializes LazyLinear) BEFORE creating the optimizer
    print("================ MODEL SUMMARY ================")
    summary(
        model_0,
        input_size=[BATCH_SIZE, 3, 224, 224],  # [batch_size, channels, height, width]
        col_names=["input_size", "output_size", "num_params"],
        device=device,
    )

    # Loss function (optionally weighted, computed from the train split only)
    if USE_CLASS_WEIGHTS:
        train_labels = np.array([labels[i] for i in train_idx])
        counts = torch.tensor(np.bincount(train_labels), dtype=torch.float)
        weights = counts.sum() / (len(counts) * counts)
        print(f"Class weights ({class_names}): {weights.tolist()}")
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model_0.parameters(), lr=LEARNING_RATE)

    # Training
    start_time = timer()
    model_0_results = train(
        model=model_0,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device,
        save_path=SAVE_PATH,
    )
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    plot_results(model_0_results)

    # Final evaluation on the test set (once, with the best checkpoint)
    model_0.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_loss, test_acc, test_f1, y_true, y_pred = test_step(
        model_0, test_loader, loss_fn, device
    )
    print(f"Test loss {test_loss:.4f} | acc {test_acc:.4f} | macro F1 {test_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names)
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()