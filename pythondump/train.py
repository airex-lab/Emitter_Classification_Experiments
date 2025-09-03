#!/usr/bin/env python3
# train.py  –  multi-GPU version
# import clearml
# from clearml import task
import os, time, json, warnings, functools, builtins
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# task.init()
# ──────────────────────────────────────────────────────────────
# 0.  Global config ––––––––––––––––––––––––––––––––––––––––––––
EMBEDDING_DIMS_TO_TEST = [32]
MARGIN           = 1.0
BASE_LR          = 1e-3      
BATCH_SIZE       = 64
EPOCHS           = 100
CLUSTER_EVERY    = 10
RESULT_DIR       = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ──────────────────────────────────────────────────────────────
# 1.  Distributed helpers –––––––––––––––––––––––––––––––––––––
def setup_distributed():
    """Initialise default process-group and bind one GPU."""
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

# ──────────────────────────────────────────────────────────────
# 2.  Data processing –––––––––––––––––––––––––––––––––––––––––
COLS = ['Name', 'PW(µs)', 'Azimuth(º)', 'Elevation(º)',
           'Power(dBm)', 'Freq(MHz)']
FEATS   = COLS[1:]
LABEL   = 'Name'

def preprocess(df, scaler=None):
    x = df[FEATS].values.astype(np.float32)
    y, uniques = pd.factorize(df[LABEL])
    if scaler is None:  # fit on train
        scaler = RobustScaler()
        x = scaler.fit_transform(x)
    else:  # transform using existing scaler
        x = scaler.transform(x)
    return x, y, {i: n for i, n in enumerate(uniques)}, scaler

class TripletPDW(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.lbl2idx = defaultdict(list)
        for i, lbl in enumerate(y): self.lbl2idx[lbl].append(i)
        if len(self.lbl2idx) < 2:
            raise ValueError("Need ≥2 classes for triplets.")

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        a = torch.from_numpy(self.x[idx])
        la = self.y[idx]
        p  = torch.from_numpy(self.x[np.random.choice(self.lbl2idx[la])])
        ln = np.random.choice([l for l in self.lbl2idx if l != la])
        n  = torch.from_numpy(self.x[np.random.choice(self.lbl2idx[ln])])
        return a, p, n

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


# ──────────────────────────────────────────────────────────────
# 3.  Model –––––––––––––––––––––––––––––––––––––––––––––––––––
class EmitterEncoder(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()
        self.net = nn.Sequential( #2 , 
            nn.Linear(in_dim,64),
            ResidualBlock(64),
            ResidualBlock(64),
            # nn.Linear(5,2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64,emb_dim)
        )
    def forward(self, x):
        return nn.functional.normalize(self.net(x), p=2, dim=1)

# ──────────────────────────────────────────────────────────────
# 4.  Clustering utils –––––––––––––––––––––––––––––––––––––––––
def clustering_acc(y_true, y_pred):
    cm  = pd.crosstab(y_pred, y_true)
    r,c = linear_sum_assignment(-cm.values)
    return cm.values[r,c].sum() / len(y_true)

def evaluate(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        emb = model(torch.tensor(x_test).cuda()).cpu().numpy()
    k   = len(np.unique(y_test))
    cid = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb)
    return clustering_acc(y_test, cid)

# ──────────────────────────────────────────────────────────────
# 5.  Main training routine –––––––––––––––––––––––––––––––––––
def main():
    rank, world = setup_distributed()
    suppress_non_main_print(rank)

    # ── Load data once per process
    df1 = pd.read_excel('/home/vishesh/github/Emitter_Classification_Experiments/dataset/set1.xls')
    df1 = df1[df1['Status'] != 'DELETE_EMITTER'][COLS]
    
    df2 = pd.read_excel('/home/vishesh/github/Emitter_Classification_Experiments/dataset/set2.xls')
    df2 = df2[df2['Status'] != 'DELETE_EMITTER'][COLS]
    
    df3 = pd.read_excel('/home/vishesh/github/Emitter_Classification_Experiments/dataset/set3.xlsx')
    df3 = df3[df3['Status'] != 'DELETE_EMITTER'][COLS]
    
    df5 = pd.read_excel('/home/vishesh/github/Emitter_Classification_Experiments/dataset/set5.xlsx')
    df5 = df5[df5['Status'] != 'DELETE_EMITTER'][COLS]
    
    df6 = pd.read_excel('/home/vishesh/github/Emitter_Classification_Experiments/dataset/set6.xlsx')
    df6 = df6[df6['Status'] != 'DELETE_EMITTER'][COLS]

    # Use set3 as test data since test_set.csv doesn't exist
    test_df = df3

    train_df = pd.concat([df1, df2, df5, df6,df3], ignore_index=True)

    x_train, y_train, _, scaler = preprocess(train_df)
    x_test,  y_test, _, _       = preprocess(test_df, scaler)

    # Dataset & distributed sampler
    ds_train  = TripletPDW(x_train, y_train)
    sampler   = DistributedSampler(ds_train, shuffle=True)
    dl_train  = DataLoader(ds_train, batch_size=BATCH_SIZE,
                           sampler=sampler, num_workers=4,
                           pin_memory=True, drop_last=True)

    # For LR scaling rule
    lr = BASE_LR * world
    if is_main(rank):
        print(f"[Rank0] Using {world} GPUs, effective LR={lr}")

    for dim in EMBEDDING_DIMS_TO_TEST:
        if is_main(rank):
            print(f"\n===== Embedding {dim} =====")

        model = EmitterEncoder(x_train.shape[1], dim).cuda()
        model = DDP(model, device_ids=[rank], output_device=rank)
        criterion = nn.TripletMarginLoss(margin=MARGIN).cuda()
        optimiser = optim.Adam(model.parameters(), lr=lr)

        for epoch in range(EPOCHS):
            sampler.set_epoch(epoch)
            model.train()
            running = 0.0
            for a,p,n in dl_train:
                # a=p=a.cuda(); n=n.cuda()
                a=a.cuda()
                p=p.cuda()
                n=n.cuda()
                optimiser.zero_grad()
                loss = criterion(model(a), model(p), model(n))
                loss.backward()
                optimiser.step()
                running += loss.item()
            avg_loss = running / len(dl_train)

            if (epoch+1) % CLUSTER_EVERY == 0 and is_main(rank):
                acc = evaluate(model.module, x_test, y_test)
                print(f"Epoch {epoch+1:3d}  loss {avg_loss:.4f}  "
                      f"test-clust-acc {acc*100:5.2f}%")

        # ---- final test (rank-0 only) ----
        if is_main(rank):
            acc = evaluate(model.module, x_test, y_test)
            out = {
                "embedding_dim": dim,
                "test_acc": acc,
                "final_loss": avg_loss,
            }
            with open(f"{RESULT_DIR}/result_dim_{dim}.json","w") as f:
                json.dump(out, f, indent=2)
            print(f"[Done] dim={dim}  acc={acc*100:.2f}%")

    if is_main(rank):
        print("\nTraining complete.")
    dist.destroy_process_group()

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
