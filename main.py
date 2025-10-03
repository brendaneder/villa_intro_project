import argparse
import glob2 as glob
import os
import random
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


print("torch.cuda.is_available():", torch.cuda.is_available())
print("Built with CUDA:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("Current device index:", torch.cuda.current_device())


n_epochs = 100

# ----------------------------
# Dataset
# ----------------------------
class NPZSirEirDataset(Dataset):
    """
    Expects .npz files each containing:
      - 'das_sir_eir': image of shape (m, n) or (1, m, n)
      - 'spheres'    : target array of shape (4,)
    """

    def __init__(self, data_dir: str, file_pattern: str = "sir_eir_outputs_*.npz", normalize: str = "minmax"):
        self.files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        if not self.files:
            self.files = sorted(glob.glob(os.path.join(data_dir, "sir_eir_outputs_*")))
        if not self.files:
            raise FileNotFoundError(f"No files found in {data_dir} matching pattern {file_pattern}")
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.files)

    def _normalize_img(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "minmax":
            x_min = np.min(x); x_max = np.max(x)
            if x_max > x_min:
                x = (x - x_min) / (x_max - x_min)
            else:
                x = np.zeros_like(x, dtype=np.float32)
            return x.astype(np.float32)
        elif self.normalize == "standard":
            mu = np.mean(x); sigma = np.std(x)
            x = (x - mu) / (sigma + 1e-8)
            return x.astype(np.float32)
        else:
            return x.astype(np.float32)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as npz:
            img = npz["das_sir_eir"]
            y = npz["spheres"].astype(np.float32)

        # Ensure (1, H, W)
        if img.ndim == 2:
            img = img[None, ...]
        elif img.ndim == 3 and img.shape[0] != 1:
            if img.shape[-1] == 1:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = np.transpose(img, (2, 0, 1))[:1, :, :]
        img = self._normalize_img(img)

        x = torch.from_numpy(img).float()      # (1, H, W)
        y = torch.from_numpy(y).float()        # (4,)
        return x, y


def peek_names(subset, k=3):
    # subset is a torch.utils.data.Subset
    ds = subset.dataset
    idxs = subset.indices if hasattr(subset, "indices") else range(len(subset))
    names = []
    for i in list(idxs)[:k]:
        names.append(os.path.basename(ds.files[i]))
    return names


# ----------------------------
# Model (CNN for regression)
# ----------------------------
class SirEirRegressor(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = self.head(x)
        return x


# ----------------------------
# Utils
# ----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset_70_15_15(ds: Dataset):
    n = len(ds)
    n_train = int(round(n * 0.70))
    n_val   = int(round(n * 0.15))
    n_test  = n - n_train - n_val
    n_test = max(n_test, 0)
    return random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))


def _ensure_bx4(preds: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Normalize shapes so both are (B, 4)
    if preds.ndim == 3 and preds.shape[-1] == 4:
        preds = preds.mean(dim=1)
    if y.ndim == 3 and y.shape[-1] == 4:
        y = y.mean(dim=1)
    if preds.ndim == 2 and preds.shape[1] != 4 and preds.shape[0] == 4:
        preds = preds.t()
    if y.ndim == 2 and y.shape[1] != 4 and y.shape[0] == 4:
        y = y.t()
    preds = preds.reshape(preds.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    assert preds.shape == y.shape and preds.shape[1] == 4, f"Mismatch: {preds.shape} vs {y.shape}"
    return preds, y


def _epoch_metrics_begin(device):
    return {
        "total_loss": 0.0,
        "count": 0,
        "per_elem_sse": torch.zeros(4, dtype=torch.float64, device=device)
    }


def _epoch_metrics_update(m, loss, preds, y, bs):
    se = (preds - y).pow(2)  # (B,4)
    m["per_elem_sse"] += se.sum(dim=0)
    m["total_loss"] += float(loss) * bs
    m["count"] += bs


def _epoch_metrics_finalize(m):
    denom = max(m["count"], 1)
    per_elem_mse = (m["per_elem_sse"] / denom).detach().cpu().numpy()
    avg_loss = m["total_loss"] / denom
    return avg_loss, per_elem_mse


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, return_preds=False):
    model.eval()
    m = _epoch_metrics_begin(device)
    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        preds = model(x)
        preds, y = _ensure_bx4(preds, y)
        loss = loss_fn(preds, y)
        _epoch_metrics_update(m, loss.item(), preds, y, x.size(0))

        if return_preds:
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())

    avg_loss, per_elem_mse = _epoch_metrics_finalize(m)

    if return_preds:
        P = torch.cat(all_preds, dim=0).numpy() if all_preds else np.zeros((0,4), dtype=np.float32)
        Y = torch.cat(all_targets, dim=0).numpy() if all_targets else np.zeros((0,4), dtype=np.float32)
        return avg_loss, per_elem_mse, P, Y
    return avg_loss, per_elem_mse


# ----------------------------
# Train
# ----------------------------
def main(n_epochs = 5):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data_nspheres_1"),
        help="Folder with sir_eir_outputs_*.npz files",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)  # EXACTLY 50 as requested
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows-safe
    parser.add_argument("--normalize", type=str, choices=["minmax", "standard", "none"], default="minmax")
    parser.add_argument("--model_out", type=str, default="sir_eir_model.pt")
    args = parser.parse_args()

    
    args.epochs = n_epochs

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # Dataset & splits
    dataset = NPZSirEirDataset(args.data_dir, normalize=args.normalize)
    train_ds, val_ds, test_ds = split_dataset_70_15_15(dataset)
    print(f"Dataset sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_mem)
    

    # Prove the two evaluate calls use different datasets
    print("val_loader.dataset is test_loader.dataset:", val_loader.dataset is test_loader.dataset)

    # Extra: ensure the test evaluate call really uses test_loader
    import inspect
    from types import FunctionType

    print("Next line should show 'test_loader' in the test evaluate call context shortly after:")
    print("Sanity: calling evaluate on val and test now with dataset fingerprints...")

    def fingerprint(loader):
        # sum of target means as a crude signature (order-independent)
        s = 0.0
        for x, y in loader:
            s += float(y.mean())
        return round(s, 6)

    print("val fp:", fingerprint(val_loader))
    print("tst fp:", fingerprint(test_loader))

    
    print("Peek train:", peek_names(train_ds))
    print("Peek val  :", peek_names(val_ds))
    print("Peek test :", peek_names(test_ds))

    val_idx = set(val_ds.indices)
    test_idx = set(test_ds.indices)
    print("val ∩ test size:", len(val_idx & test_idx))
    print("val == test:", val_idx == test_idx)


    # Model / Optim / Loss
    model = SirEirRegressor(in_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Storage for curves (TEST metrics each epoch)
    epochs = list(range(1, args.epochs + 1))
    test_total_mse_hist: List[float] = []
    test_bias2_hist: List[float] = []
    test_var_hist: List[float] = []
    test_rmse_per_elem_hist: List[np.ndarray] = []  # each entry shape (4,)
    test_total_rmse_hist: List[float] = []

    best_val = float("inf")  # we still save best-by-val, but do NOT early stop

    for epoch in epochs:
        # ---- Train ----
        model.train()
        m_tr = _epoch_metrics_begin(device)
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                preds = model(x)
                preds, y = _ensure_bx4(preds, y)
                loss = loss_fn(preds, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            _epoch_metrics_update(m_tr, loss.item(), preds, y, x.size(0))
        tr_loss, tr_mse_vec = _epoch_metrics_finalize(m_tr)

        # ---- Val ----
        va_loss, va_mse_vec = evaluate(model, val_loader, loss_fn, device)
        

        # Save best-by-val (no early stop)
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            torch.save({"model_state": model.state_dict(),
                        "best_val_mse": best_val}, args.model_out)
            print(f"  ↳ saved new best to {args.model_out}")

        # ---- Test (strictly separate) ----
        te_loss, te_mse_vec, P, Y = evaluate(model, test_loader, loss_fn, device, return_preds=True)

        # Reporting
        print(
            f"Epoch {epoch:03d}\n"
            f"  TRAIN  MSE: {tr_loss:.6f} | per-elem: [{tr_mse_vec[0]:.6f}, {tr_mse_vec[1]:.6f}, {tr_mse_vec[2]:.6f}, {tr_mse_vec[3]:.6f}]\n"
            f"  VAL    MSE: {va_loss:.6f} | per-elem: [{va_mse_vec[0]:.6f}, {va_mse_vec[1]:.6f}, {va_mse_vec[2]:.6f}, {va_mse_vec[3]:.6f}]\n"
            f"  TEST   MSE: {te_loss:.6f} | per-elem: [{te_mse_vec[0]:.6f}, {te_mse_vec[1]:.6f}, {te_mse_vec[2]:.6f}, {te_mse_vec[3]:.6f}]"
        )

        # ---- Curves (computed on TEST set per epoch) ----
        # Total MSE as mean of element-wise MSE:
        total_mse = float(np.mean(te_mse_vec))
        test_total_mse_hist.append(total_mse)

        if P.size > 0:
            # Bias per element: mean(P - Y)
            bias = (P - Y).mean(axis=0)            # shape (4,)
            bias2 = float(np.mean(bias**2))        # scalar Bias^2 averaged over 4 elements
            test_bias2_hist.append(bias2)

            # Variance of predictions per element:
            var = P.var(axis=0, ddof=0)            # shape (4,)
            var_mean = float(np.mean(var))         # scalar variance averaged over 4 elements
            test_var_hist.append(var_mean)

            # RMSE per element + total RMSE:
            rmse_elem = np.sqrt(te_mse_vec)        # shape (4,)
            test_rmse_per_elem_hist.append(rmse_elem)
            test_total_rmse_hist.append(float(np.sqrt(total_mse)))
        else:
            # Edge case: empty test set
            test_bias2_hist.append(0.0)
            test_var_hist.append(0.0)
            test_rmse_per_elem_hist.append(np.zeros(4, dtype=np.float32))
            test_total_rmse_hist.append(0.0)

    # ----------------------------
    # Plots (computed on TEST over epochs)
    # ----------------------------
    # 1) Bias^2, Variance, Total Error (MSE)
    plt.figure()
    plt.plot(epochs, test_bias2_hist, label="Bias^2")
    plt.plot(epochs, test_var_hist, label="Variance")
    plt.plot(epochs, test_total_mse_hist, label="Total Error (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.yscale("log")
    plt.title("Bias^2, Variance, and Total Error (Test) vs. Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("bias_variance_mse_over_epochs.png", dpi=150)
    plt.close()

    # 2) RMSE for x, y, z, R and total RMSE
    rmse_matrix = np.vstack(test_rmse_per_elem_hist) if len(test_rmse_per_elem_hist) > 0 else np.zeros((len(epochs), 4))
    plt.figure()
    if rmse_matrix.shape[0] > 0:
        plt.plot(epochs, rmse_matrix[:, 0], label="x RMSE")
        plt.plot(epochs, rmse_matrix[:, 1], label="y RMSE")
        plt.plot(epochs, rmse_matrix[:, 2], label="z RMSE")
        plt.plot(epochs, rmse_matrix[:, 3], label="R RMSE")
    plt.plot(epochs, test_total_rmse_hist, label="Total RMSE", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.yscale("log")
    plt.title("Per-element and Total RMSE (Test) vs. Epoch")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("rmse_over_epochs.png", dpi=150)
    plt.close()

    print("Saved plots: 'bias_variance_mse_over_epochs.png' and 'rmse_over_epochs.png'")


if __name__ == "__main__":
    main(n_epochs)


