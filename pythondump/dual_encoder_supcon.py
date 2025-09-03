#!/usr/bin/env python3
# dual_encoder.py — multi-GPU bi-encoder with Supervised Contrastive Loss (SupCon)

import os, json, warnings, functools, builtins
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# ──────────────────────────────────────────────────────────────
# 0.  Global config –––––––––––––––––––––––––––––––––––––––––───
EMBEDDING_DIMS_TO_TEST = [32, 16, 64, 8, 4, 2]
BASE_LR        = 1e-3
BATCH_SIZE     = 64
EPOCHS         = 100
CLUSTER_EVERY  = 10
RESULT_DIR     = "results"
NUM_WORKERS    = 4
TEMP           = 0.07      # SupCon temperature
AMP            = True      # mixed precision
os.makedirs(RESULT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ──────────────────────────────────────────────────────────────
# 1.  Distributed helpers –––––––––––––––––––––––––─────────────
def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size

def is_main(rank: int) -> bool:
    return rank == 0

def suppress_non_main_print(rank: int):
    if not is_main(rank):
        builtins.print = functools.partial(lambda *a, **k: None)

@torch.no_grad()
def concat_all_gather(t: torch.Tensor) -> torch.Tensor:
    """
    Gathers tensors from all ranks and concatenates on dim=0.
    This version is used only on detached inputs so it won't carry a graph.
    """
    world = dist.get_world_size()
    tensors_gather = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(tensors_gather, t.contiguous())
    return torch.cat(tensors_gather, dim=0)

# ──────────────────────────────────────────────────────────────
# 2.  Data processing –––––––––––––––––─────────────────────────
COLS  = ['Name', 'PW(µs)', 'Azimuth(º)', 'Elevation(º)', 'Power(dBm)', 'Freq(MHz)']
FEATS = COLS[1:]
LABEL = 'Name'

def preprocess(df, scaler=None):
    x = df[FEATS].values.astype(np.float32)
    y, uniques = pd.factorize(df[LABEL])
    if scaler is None:
        scaler = RobustScaler()
        x = scaler.fit_transform(x)
    else:
        x = scaler.transform(x)
    return x, y, {i: n for i, n in enumerate(uniques)}, scaler

class EmitterDataset(Dataset):
    """
    Returns (feature, label) for each sample.
    Positives are determined by labels in the batch for SupCon.
    """
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.lbl2idx = defaultdict(list)
        for i, lbl in enumerate(y):
            self.lbl2idx[lbl].append(i)
        enough = [l for l, idxs in self.lbl2idx.items() if len(idxs) >= 2]
        if len(enough) < 2:
            raise ValueError("Need ≥2 labels with ≥2 samples each for contrastive learning.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])

# ──────────────────────────────────────────────────────────────
# 3.  Model –––––––––––––––––───────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.relu(out + residual)

class EmitterEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Linear(64, emb_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return nn.functional.normalize(z, p=2, dim=1)

# ──────────────────────────────────────────────────────────────
# 4.  Supervised Contrastive Loss –––––––––─────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_local, y_local, z_global, y_global):
        """
        Computes SupCon loss on local anchors vs global positives/negatives.
        z_local: local normalized embeddings [B, D] (with grad)
        y_local: local labels [B]
        z_global: global normalized embeddings [B*world, D] (detached)
        y_global: global labels [B*world]
        """
        B = z_local.size(0)
        device = z_local.device

        # Expand labels for mask
        y_local_exp = y_local.view(-1, 1)  # [B, 1]
        y_global_exp = y_global.view(1, -1)  # [1, B*world]

        # Mask of positives (same label, exclude self)
        mask = (y_local_exp == y_global_exp).float().to(device)  # [B, B*world]
        self_mask = torch.eye(B, device=device)
        self_mask = torch.cat([self_mask] * dist.get_world_size(), dim=1)  # [B, B*world]
        mask = mask * (1 - self_mask)  # exclude self

        # Logits: local anchors vs global embeddings
        logits = (z_local @ z_global.T) / self.temperature  # [B, B*world]

        # For stability, subtract max per row
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max

        # Positives mask for numerator
        pos_logits = torch.exp(logits) * mask
        pos_sum = pos_logits.sum(1, keepdim=True)

        # Denominator: all exp logits (mask out self already handled)
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum(1, keepdim=True)

        # Log prob, averaged over positives
        log_prob = torch.log(pos_sum / denom + 1e-9)  # avoid div by zero
        mean_log_prob_pos = log_prob.sum(1) / (mask.sum(1) + 1e-9)

        # Loss: negative mean log prob
        loss = -mean_log_prob_pos
        valid = (mask.sum(1) > 0)  # ignore anchors with no positives
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)
        return loss[valid].mean()

# ──────────────────────────────────────────────────────────────
# 5.  Evaluation –──────────────────────────────────────────────
def clustering_acc(y_true, y_pred):
    cm  = pd.crosstab(y_pred, y_true)
    r, c = linear_sum_assignment(-cm.values)
    return cm.values[r, c].sum() / len(y_true)

@torch.no_grad()
def evaluate(model, x_test, y_test):
    model.eval()
    emb = model(torch.tensor(x_test).cuda()).cpu().numpy()
    k   = len(np.unique(y_test))
    cid = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb)
    return clustering_acc(y_test, cid)

# ──────────────────────────────────────────────────────────────
# 6.  Main training routine –───────────────────────────────────
def main():
    rank, world = setup_distributed()
    suppress_non_main_print(rank)

    try:
        # ── Load data once per process
        df1 = pd.read_excel('../dataset/set1.xls')
        df1 = df1[df1['Status'] != 'DELETE_EMITTER'][COLS]

        df2 = pd.read_excel('../dataset/set2.xls')
        df2 = df2[df2['Status'] != 'DELETE_EMITTER'][COLS]

        df3 = pd.read_excel('../dataset/set3.xlsx')
        df3 = df3[df3['Status'] != 'DELETE_EMITTER'][COLS]

        df5 = pd.read_excel('../dataset/set5.xlsx')
        df5 = df5[df5['Status'] != 'DELETE_EMITTER'][COLS]

        df6 = pd.read_excel('../dataset/set6.xlsx')
        df6 = df6[df6['Status'] != 'DELETE_EMITTER'][COLS]

        train_df = pd.concat([df1, df2, df5, df6], ignore_index=True)
        test_df  = df3

        x_train, y_train, _, scaler = preprocess(train_df)
        x_test,  y_test,  _, _      = preprocess(test_df, scaler)

        # Dataset & distributed sampler
        ds_train = EmitterDataset(x_train, y_train)
        sampler  = DistributedSampler(ds_train, shuffle=True)
        dl_train = DataLoader(
            ds_train,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        # LR scaling rule
        lr = BASE_LR * world
        if is_main(rank):
            print(f"[Rank0] Using {world} GPUs, effective LR={lr}")

        scaler_amp = torch.amp.GradScaler('cuda', enabled=AMP)
        criterion = SupConLoss(temperature=TEMP)

        in_dim = x_train.shape[1]

        for dim in EMBEDDING_DIMS_TO_TEST:
            if is_main(rank):
                print(f"\n===== Embedding {dim} =====")

            model = EmitterEncoder(in_dim, dim).cuda()
            model = DDP(model, device_ids=[rank], output_device=rank)

            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(EPOCHS):
                sampler.set_epoch(epoch)
                model.train()
                running = 0.0

                for batch in dl_train:
                    x, y = batch
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    # Forward with AMP
                    with torch.amp.autocast('cuda', enabled=AMP):
                        z_local = model(x)  # [B, D], with grad

                    # Gather GLOBAL detached for negatives/positives
                    z_global = concat_all_gather(z_local.detach())
                    y_global = concat_all_gather(y)

                    # Compute loss (local anchors vs global)
                    with torch.amp.autocast('cuda', enabled=AMP):
                        loss = criterion(z_local, y, z_global, y_global)

                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(optimizer)
                    scaler_amp.update()

                    running += float(loss.detach().item())

                avg_loss = running / len(dl_train)

                if (epoch + 1) % CLUSTER_EVERY == 0 and is_main(rank):
                    acc = evaluate(model.module, x_test, y_test)
                    print(f"Epoch {epoch+1:3d}  loss {avg_loss:.4f}  test-clust-acc {acc*100:5.2f}%")

            if is_main(rank):
                acc = evaluate(model.module, x_test, y_test)
                out = {
                    "embedding_dim": dim,
                    "test_acc": acc,
                    "final_loss": avg_loss,
                    "temperature": TEMP,
                }
                with open(f"{RESULT_DIR}/result_dim_{dim}.json", "w") as f:
                    json.dump(out, f, indent=2)
                print(f"[Done] dim={dim}  acc={acc*100:.2f}%")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
