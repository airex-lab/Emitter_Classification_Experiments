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
import torch.utils.data as tud
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.optimize import linear_sum_assignment

# Optional HDBSCAN (graceful fallback if not installed), use dynamic import to avoid linter error
import importlib
_hdbscan_spec = importlib.util.find_spec("hdbscan")
if _hdbscan_spec is not None:
    HDBSCAN = importlib.import_module("hdbscan").HDBSCAN  # type: ignore[attr-defined]
    HAS_HDBSCAN = True
else:
    HDBSCAN = None  # type: ignore[assignment]
    HAS_HDBSCAN = False


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
FEATS = ['PW(µs)', 'Azimuth(º)', 'Elevation(º)', 'Power(dBm)', 'Freq(MHz)']
LABEL = 'Name'

RAW_TO_TRAIN_COLS = {
    'PW(usec)': 'PW(µs)',
    'Azimuth(deg)': 'Azimuth(º)',
    'Elevation/ANT.Power.2': 'Elevation(º)',
    'Power ': 'Power(dBm)',
    'Frequency(MHz)': 'Freq(MHz)',
}

def preprocess(df, scaler=None):
    x = df[FEATS].values.astype(np.float32)
    y, uniques = pd.factorize(df[LABEL])
    if scaler is None:  # fit on train
        scaler = RobustScaler()
        x = scaler.fit_transform(x)
    else:  # transform using existing scaler
        x = scaler.transform(x)
    return x, y, {i: n for i, n in enumerate(uniques)}, scaler


def preprocess_infer(df, scaler):
    # Map raw columns to train columns, keep only FEATS
    df = df.rename(columns=RAW_TO_TRAIN_COLS)
    missing = [c for c in FEATS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in inference df: {missing}. Got: {df.columns.tolist()}")
    x = df[FEATS].values.astype(np.float32)
    if scaler is not None:
        x = scaler.transform(x)
    return x


class TripletPDW(tud.Dataset):
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
        # Deeper MLP with residual blocks
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            ResidualBlock(64),
            nn.Linear(64, emb_dim)
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
# 4b. Visualization (test_df only; no labels) –––––––––––––––––
def _save_cluster_stats_and_assignments(features_df, labels, out_dir, method_name):
    """Save per-cluster descriptive stats and per-PDW assignments to CSVs."""
    import pandas as _pd
    # Ensure DataFrame with expected columns
    if isinstance(features_df, np.ndarray):
        df = _pd.DataFrame(features_df, columns=FEATS)
    else:
        df = features_df.copy()
        df = df[FEATS]

    out_dir = os.path.join(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df_assign = df.copy()
    df_assign['cluster'] = labels
    assign_path = os.path.join(out_dir, f'{method_name}_assignments.csv')
    df_assign.to_csv(assign_path, index=False)

    rows = []
    for cid, g in df_assign.groupby('cluster', sort=True):
        stats_row = {
            'cluster_id': int(cid),
            'is_noise': bool(cid == -1),
            'count': int(len(g)),
            'freq_unique': int(g['Freq(MHz)'].nunique()) if 'Freq(MHz)' in g else int(len(g))
        }
        for col in FEATS:
            s = g[col]
            stats_row[f'{col}__mean'] = float(s.mean())
            stats_row[f'{col}__median'] = float(s.median())
            stats_row[f'{col}__std'] = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            stats_row[f'{col}__min'] = float(s.min())
            stats_row[f'{col}__max'] = float(s.max())
            stats_row[f'{col}__q25'] = float(s.quantile(0.25))
            stats_row[f'{col}__q75'] = float(s.quantile(0.75))
        rows.append(stats_row)

    stats_df = _pd.DataFrame(rows)
    stats_path = os.path.join(out_dir, f'{method_name}_cluster_stats.csv')
    stats_df.to_csv(stats_path, index=False)


def visualize_test_embeddings_elbow_kmeans(model, x_test, out_dir, raw_features_df=None):
    """
    - Extract embeddings for test_df.
    - Choose K via Elbow (inertia/SSE) over a sweep of K.
    - Visualize KMeans clusters on 2D PCA and 2D t-SNE projections.
    Saves: elbow_inertia.png, test_pca_kmeans.png, test_tsne_kmeans.png
    Also saves per-method cluster stats and assignments as CSV if raw_features_df is provided:
    - kmeans_cluster_stats.csv, kmeans_assignments.csv
    - dbscan_cluster_stats.csv, dbscan_assignments.csv
    - hdbscan_cluster_stats.csv, hdbscan_assignments.csv (if hdbscan available)
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        emb = model(torch.tensor(x_test).cuda()).cpu().numpy()

    # Elbow method: inertia vs K
    n = emb.shape[0]  # number of samples
    k_min = 1
    k_max = min(12, max(2, n))  # cap upper bound; at least 2 if enough points
    ks = list(range(k_min, k_max+1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(emb)
        inertias.append(km.inertia_)

    # Plot inertia curve
    plt.figure(figsize=(6,5))
    plt.plot(ks, inertias, marker='o')
    plt.xticks(ks)
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia (SSE)')
    plt.title('Elbow Method (KMeans on test embeddings)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'elbow_inertia.png'), dpi=150)
    plt.close()

    # Heuristic elbow selection: "knee" via maximum second derivative discrete approx
    # If too few points, fallback to K=2 or K=1 depending on availability
    k_selected = 2 if len(ks) >= 2 else 1
    if len(ks) >= 3:
        y = np.array(inertias, dtype=float)
        x = np.array(ks, dtype=float)
        # Normalize to [0,1] to balance scales
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
        # Second difference magnitude as elbow proxy
        sec_diff = np.abs(np.diff(y_n, n=2))
        # sec_diff indexes correspond to ks[0+2:] -> ks[2:]
        idx = np.argmax(sec_diff)
        k_selected = ks[idx + 2 - 1]  # approximate elbow near second-diff peak
    elif len(ks) == 2:
        # Pick the larger K if inertia drop is significant, else 1
        drop = inertias[0] - inertias[1]
        k_selected = 2 if drop > 0 else 1

    # Fit KMeans with selected K
    km_final = KMeans(n_clusters=k_selected, n_init=10, random_state=42).fit(emb)
    y_km = km_final.labels_
    centers = km_final.cluster_centers_

    # 2D reductions
    pca = PCA(n_components=2, random_state=42)
    pca2 = pca.fit_transform(emb)
    pca_centers = pca.transform(centers)

    # t-SNE (perplexity must be < n_samples; use safe bound)
    perplexity = max(5, min(30, max(5, (n - 1) // 3)))
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', learning_rate='auto', random_state=42)
    # For consistent center placement, fit t-SNE on combined data (emb + centers)
    all_emb = np.vstack([emb, centers])
    tsne_all = tsne.fit_transform(all_emb)
    tsne_points = tsne_all[:-centers.shape[0]]
    tsne_centers = tsne_all[-centers.shape[0]:]

    # Plot PCA clusters
    plt.figure(figsize=(6,5))
    plt.scatter(pca2[:,0], pca2[:,1], c=y_km, cmap='tab20', s=12, alpha=0.9)
    plt.scatter(pca_centers[:,0], pca_centers[:,1], c='black', s=120, marker='X', alpha=0.9, edgecolors='white', linewidths=1.0)
    plt.title(f'KMeans (K={k_selected}) on test embeddings – PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_pca_kmeans.png'), dpi=150)
    plt.close()

    # Plot t-SNE clusters
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_points[:,0], tsne_points[:,1], c=y_km, cmap='tab20', s=12, alpha=0.9)
    plt.scatter(tsne_centers[:,0], tsne_centers[:,1], c='black', s=120, marker='X', alpha=0.9, edgecolors='white', linewidths=1.0)
    plt.title(f'KMeans (K={k_selected}) on test embeddings – t-SNE')
    plt.xlabel('dim-1')
    plt.ylabel('dim-2')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_tsne_kmeans.png'), dpi=150)
    plt.close()

    # Save stats/assignments for KMeans
    if raw_features_df is not None:
        _save_cluster_stats_and_assignments(raw_features_df, y_km, out_dir, 'kmeans')

    # ─────────────────────────────────────────────────────
    # Additional clustering: DBSCAN and HDBSCAN (if avail)
    # ─────────────────────────────────────────────────────
    def choose_dbscan_eps(embeddings):
        # Simple heuristic sweep; pick first eps giving >=2 clusters and not all noise
        for eps in [0.2, 0.3, 0.4, 0.5, 0.6]:
            labels = DBSCAN(eps=eps, min_samples=5).fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2:
                return eps, labels
        labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(embeddings)
        return 0.5, labels

    # DBSCAN
    eps_sel, y_db = choose_dbscan_eps(emb)

    # Plot PCA clusters (DBSCAN)
    plt.figure(figsize=(6,5))
    plt.scatter(pca2[:,0], pca2[:,1], c=y_db, cmap='tab20', s=12, alpha=0.9)
    plt.title(f'DBSCAN (eps={eps_sel:.2f}) on test embeddings – PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_pca_dbscan.png'), dpi=150)
    plt.close()

    # Plot t-SNE clusters (DBSCAN)
    plt.figure(figsize=(6,5))
    plt.scatter(tsne_points[:,0], tsne_points[:,1], c=y_db, cmap='tab20', s=12, alpha=0.9)
    plt.title(f'DBSCAN (eps={eps_sel:.2f}) on test embeddings – t-SNE')
    plt.xlabel('dim-1')
    plt.ylabel('dim-2')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'test_tsne_dbscan.png'), dpi=150)
    plt.close()

    # Save stats/assignments for DBSCAN
    if raw_features_df is not None:
        _save_cluster_stats_and_assignments(raw_features_df, y_db, out_dir, 'dbscan')

    # HDBSCAN (if installed)
    if HAS_HDBSCAN:
        # Try a couple of min_cluster_size values; pick the first that yields >=2 clusters
        mcs_candidates = [max(5, n//20), max(5, n//10), 10]
        y_hdb = None
        mcs_sel = None
        for mcs in mcs_candidates:
            y_try = HDBSCAN(min_cluster_size=mcs).fit_predict(emb)
            n_clusters = len(set(y_try)) - (1 if -1 in y_try else 0)
            if n_clusters >= 2:
                y_hdb = y_try
                mcs_sel = mcs
                break
        if y_hdb is None:
            mcs_sel = 10
            y_hdb = HDBSCAN(min_cluster_size=mcs_sel).fit_predict(emb)

        # PCA plot (HDBSCAN)
        plt.figure(figsize=(6,5))
        plt.scatter(pca2[:,0], pca2[:,1], c=y_hdb, cmap='tab20', s=12, alpha=0.9)
        plt.title(f'HDBSCAN (min_cluster_size={mcs_sel}) on test embeddings – PCA')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'test_pca_hdbscan.png'), dpi=150)
        plt.close()

        # t-SNE plot (HDBSCAN)
        plt.figure(figsize=(6,5))
        plt.scatter(tsne_points[:,0], tsne_points[:,1], c=y_hdb, cmap='tab20', s=12, alpha=0.9)
        plt.title(f'HDBSCAN (min_cluster_size={mcs_sel}) on test embeddings – t-SNE')
        plt.xlabel('dim-1')
        plt.ylabel('dim-2')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'test_tsne_hdbscan.png'), dpi=150)
        plt.close()
        # Save stats/assignments for HDBSCAN
        if raw_features_df is not None:
            _save_cluster_stats_and_assignments(raw_features_df, y_hdb, out_dir, 'hdbscan')
    else:
        with open(os.path.join(out_dir, 'hdbscan_unavailable.txt'), 'w') as f:
            f.write('Install the hdbscan package to enable HDBSCAN clustering/plots: pip install hdbscan\n')


# ──────────────────────────────────────────────────────────────
# 5.  Main training routine –––––––––––––––––––––––––––––––––––
def is_main_process():
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True

def main():
    rank, world = setup_distributed()
    suppress_non_main_print(rank)


    # ── Load data once per process
    df1 = pd.read_excel('set1.xls')
    df1 = df1[df1['Status'] != 'DELETE_EMITTER'][COLS]
    
    df2 = pd.read_excel('set2.xls')
    df2 = df2[df2['Status'] != 'DELETE_EMITTER'][COLS]
    
    df3 = pd.read_excel('set3.xlsx')
    df3 = df3[df3['Status'] != 'DELETE_EMITTER'][COLS]
    
    df5 = pd.read_excel('set5.xlsx')
    df5 = df5[df5['Status'] != 'DELETE_EMITTER'][COLS]
    
    df6 = pd.read_excel('set6.xlsx')
    df6 = df6[df6['Status'] != 'DELETE_EMITTER'][COLS]

    col=['PW(usec)','Azimuth(deg)','Power ','Frequency(MHz)','Elevation/ANT.Power.2']
    
    # Load all three test datasets
    test_datasets = {
        's3cleaned': pd.read_csv('s3cleaned.csv')[col],
        's6cleaned': pd.read_csv('s6cleaned.csv')[col], 
        'raw1_cleaned': pd.read_csv('raw1_cleaned.csv')[col]
    }

    train_df = pd.concat([df1, df2, df5, df6,df3], ignore_index=True)

    x_train, y_train, _, scaler = preprocess(train_df)

    # Column mapping from raw data to processed format
    column_mapping = {
        'PW(usec)': 'PW(µs)',
        'Azimuth(deg)': 'Azimuth(º)', 
        'Power ': 'Power(dBm)',
        'Frequency(MHz)': 'Freq(MHz)',
        'Elevation/ANT.Power.2': 'Elevation(º)'
    }
    
    # Process all test datasets for clustering (no labels needed)
    x_test_datasets = {}
    raw_features_by_name = {}
    for name, test_df in test_datasets.items():
        # Rename columns to match training format
        test_df_processed = test_df.rename(columns=column_mapping)
        # Keep a copy of raw features with expected FEATS
        raw_features_by_name[name] = test_df_processed[FEATS].copy()
        # Process for clustering (scaled)
        x_test_datasets[name] = preprocess_infer(test_df_processed, scaler)


    # Dataset & distributed sampler
    ds_train  = TripletPDW(x_train, y_train)
    sampler   = tud.DistributedSampler(ds_train, shuffle=True)
    dl_train  = tud.DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        sampler=sampler,   # DistributedSampler or RandomSampler
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
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
                print(f"Epoch {epoch+1:3d}  loss {avg_loss:.4f}")


        # ---- final test (rank-0 only) ----
        if is_main(rank):
            # Save the trained model
            model_save_path = f"{RESULT_DIR}/trained_model_dim_{dim}.pth"
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'embedding_dim': dim,
                'final_loss': avg_loss,
                'scaler': scaler,
                'feature_names': FEATS,
                'input_dim': x_train.shape[1]
            }, model_save_path)
            print(f"Model saved to: {model_save_path}")
            
            out = {
                "embedding_dim": dim,
                "final_loss": avg_loss,
                "model_path": model_save_path,
            }
            with open(f"{RESULT_DIR}/result_dim_{dim}.json","w") as f:
                json.dump(out, f, indent=2)
            print(f"[Done] dim={dim}  final_loss={avg_loss:.4f}")


    # === Visualization on all test datasets (rank-0 only) ===
    if is_main(rank):
        for dataset_name, x_test in x_test_datasets.items():
            try:
                # Create separate result directory for each dataset
                dataset_result_dir = f"{RESULT_DIR}/{dataset_name}"
                os.makedirs(dataset_result_dir, exist_ok=True)
                
                # Generate visualizations for this dataset
                visualize_test_embeddings_elbow_kmeans(
                    model.module,
                    x_test,
                    dataset_result_dir,
                    raw_features_df=raw_features_by_name.get(dataset_name)
                )
                print(f"Saved {dataset_name} embeddings visualizations to {dataset_result_dir}/")
            except Exception as e:
                print(f"Visualization failed for {dataset_name}: {e}")

    # Clean DDP shutdown
    try:
        if dist.is_initialized():
            dist.barrier()
            if is_main(rank):
                print("Done.")
            dist.destroy_process_group()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
