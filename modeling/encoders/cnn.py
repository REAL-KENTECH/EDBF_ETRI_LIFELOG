"""cnn.py — dilated 1D-CNN with per-subject embedding over the sleep-window tensor.

Model definition plus the training and inference primitives. Consumes cache/sleep_window_tensor.npz
and produces 7 per-label OOF/sub probabilities.

  PyramidSleepCNN     multi-resolution branches with mask-aware pooling
  train_cnn           optimization (learning rate, schedule, early stop)
  normalize_per_fold  mask-aware input normalization
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # required for deterministic CUDA matmul
import numpy as np
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PyramidSleepCNN(nn.Module):
    """Multi-resolution sleep-tensor CNN, one end-to-end module.

    Parallel branches see the tensor at 5/10/15 min through mask-aware average pooling (factors
    1, 2, 3); the head fuses them jointly. Training is seeded and cuDNN-deterministic, but the GPU
    draw differs across machines, so a rebuild does not byte-reproduce cache/view_sleep_cnn.npz."""

    def __init__(self, n_ch, n_subj, n_lab=7, factors=(1, 2, 3), w=20, sd=12, dropout=0.3):
        super().__init__()
        self.factors = factors
        k = int(os.environ.get('CNN_KERNEL', 7)); p = (k - 1) // 2
        self.branch = nn.ModuleList([
            nn.Sequential(nn.Conv1d(n_ch, w, k, padding=p), nn.BatchNorm1d(w), nn.ReLU(),
                          nn.Conv1d(w, w, k, padding=p * 2, dilation=2), nn.BatchNorm1d(w), nn.ReLU())
            for _ in factors])
        self.d = nn.Dropout(dropout)
        self.subj = nn.Embedding(n_subj, sd)
        self.head = nn.Sequential(nn.Linear(w * len(factors) + sd, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, n_lab))

    def forward(self, x, mask, subj):
        import torch.nn.functional as F
        feats = []
        for f, br in zip(self.factors, self.branch):
            if f > 1:
                xm = F.avg_pool1d(x * mask, f); mm = F.avg_pool1d(mask, f)
                xd = xm / mm.clamp(min=1e-6); md = (mm > 0).float()
            else:
                xd, md = x * mask, mask
            cw = md.mean(dim=1, keepdim=True)
            h = self.d(br(xd))
            feats.append((h * cw).sum(dim=2) / cw.sum(dim=2).clamp(min=1e-6))
        return self.head(torch.cat(feats + [self.subj(subj)], dim=1))


def normalize_per_fold(X_tr, M_tr, *Xs):
    """Z-score normalize sensor channels using train-fold statistics (leak-safe)."""
    eps = 1e-06
    means = np.zeros(X_tr.shape[1])
    stds = np.ones(X_tr.shape[1])
    for c in range(X_tr.shape[1]):
        v = X_tr[:, c, :][M_tr[:, c, :] > 0]
        if len(v) > 0:
            means[c] = v.mean()
            stds[c] = v.std() + eps

    def app(X):
        Xn = X.copy()
        for c in range(X.shape[1]):
            Xn[:, c, :] = (X[:, c, :] - means[c]) / stds[c]
        return Xn
    return tuple((app(x) for x in (X_tr,) + Xs))


def train_cnn(X_tr, M_tr, s_tr, y_tr, X_va, M_va, s_va, y_va, n_ch, n_subj, seed=42, max_epochs=300, patience=50, lr=0.0005, batch_size=32, model_cls=PyramidSleepCNN):
    """Train the sleep-tensor CNN (PyramidSleepCNN — the sole, multi-resolution encoder) on one fold
    with early stopping on validation BCE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # deterministic CNN: fixed seed + cuDNN determinism
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    model = model_cls(n_ch=n_ch, n_subj=n_subj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    to_t = lambda a: torch.from_numpy(a).float().to(device)
    (Xtr, Mtr, ytr) = (to_t(X_tr), to_t(M_tr), to_t(y_tr))
    str_ = torch.from_numpy(s_tr).long().to(device)
    (Xva, Mva, yva) = (to_t(X_va), to_t(M_va), to_t(y_va))
    sva = torch.from_numpy(s_va).long().to(device)
    n = len(X_tr)
    best_val = float('inf')
    best_state = {k: v.clone() for (k, v) in model.state_dict().items()}
    bad = 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xtr[idx], Mtr[idx], str_[idx])
            loss = loss_fn(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(Xva, Mva, sva), yva).item()
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.clone() for (k, v) in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


def predict_cnn(model, X, M, s, batch_size=128):
    """Run trained CNN inference and return per-label probabilities."""
    model.eval()
    Xt = torch.from_numpy(X).float().to(device)
    Mt = torch.from_numpy(M).float().to(device)
    st = torch.from_numpy(s).long().to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(torch.sigmoid(model(Xt[i:i + batch_size], Mt[i:i + batch_size], st[i:i + batch_size])).cpu().numpy())
    return np.concatenate(out, axis=0)
