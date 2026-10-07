"""Phase 3: neural network (MLP). Same train/validation rows, columns and scores as Phase 2. Test is never touched.

Every preprocessing statistic is computed from TRAIN rows only (fit_prep), then applied unchanged (transform).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.lgbm import feature_names, no_skill  # same column list and no-skill line as LightGBM
from src.split import split_by_time

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
THREADS = 4  # fixed, as in Phase 2: the thread count changes rounding, so it must not vary
# One configuration, fixed before any validation score was seen. Not tuned (that is Phase 4).
HIDDEN = (256, 128)
DROPOUT = 0.3
LR = 1e-3
BATCH = 1024
# Stopping rule reused from Phase 2: validation ROC-AUC, best one kept, patience (here in epochs, not trees).
MAX_EPOCHS = 50
PATIENCE = 5
MIN_COUNT = 10       # a label needs this many train rows to get its own embedding slot
MAX_EMB = 16         # largest embedding size
CLIP = 5.0           # scaled values are cut to [-5, 5]
UNKNOWN, MISSING = 0, 1  # embedding slots; vocabulary labels start at 2
TRAIN_SAMPLE = 50_000    # train rows scored to compare with validation


def slog(x):
    """Signed log: pulls huge values in, keeps the sign. Has no fitted numbers."""
    return np.sign(x) * np.log1p(np.abs(x))


def fit_prep(tr, feats):
    """Every statistic the network uses, computed from TRAIN rows only."""
    cat = [c for c in feats if str(tr[c].dtype) == "category"]
    num = [c for c in feats if c not in cat]
    prep = {"num": num, "cat": cat, "median": {}, "mean": {}, "std": {}, "flag": [], "vocab": {}}
    for c in num:
        v = slog(tr[c].to_numpy("float64"))
        miss = np.isnan(v)
        med = float(np.median(v[~miss])) if (~miss).any() else 0.0
        v[miss] = med
        std = float(v.std())
        prep["median"][c], prep["mean"][c], prep["std"][c] = med, float(v.mean()), std if std > 1e-6 else 1.0
        if miss.any():
            prep["flag"].append(c)
    for c in cat:
        # counted in train rows; the dtype's category list covers the whole file, so it is NOT used
        counts = tr[c].value_counts()
        prep["vocab"][c] = sorted(str(k) for k, n in counts.items() if n >= MIN_COUNT)
    return prep


def transform(rows, prep):
    """Apply the TRAIN statistics to any rows. Never fits anything. Returns (numbers, category codes)."""
    num, flag_pos = prep["num"], {c: len(prep["num"]) + k for k, c in enumerate(prep["flag"])}
    x = np.empty((len(rows), len(num) + len(flag_pos)), dtype=np.float32)
    for j, c in enumerate(num):
        v = slog(rows[c].to_numpy("float64"))
        miss = np.isnan(v)
        v[miss] = prep["median"][c]
        x[:, j] = np.clip((v - prep["mean"][c]) / prep["std"][c], -CLIP, CLIP)  # clip also turns +-inf into +-5
        if c in flag_pos:
            x[:, flag_pos[c]] = miss
    codes = np.empty((len(rows), len(prep["cat"])), dtype=np.int64)
    for j, c in enumerate(prep["cat"]):
        slot = {lab: i + 2 for i, lab in enumerate(prep["vocab"][c])}
        # one slot per category of this table, plus MISSING last: pandas codes a blank as -1, which picks the last entry
        lookup = np.array([slot.get(str(k), UNKNOWN) for k in rows[c].cat.categories] + [MISSING])
        codes[:, j] = lookup[rows[c].cat.codes.to_numpy()]
    return torch.from_numpy(x), torch.from_numpy(codes)


def prepare(df, n_train=None, seed=SEED):
    """Split by time, fit statistics on TRAIN rows only, apply them to train and validation.

    Returns (prep, (x_num, x_cat, y) for train, (x_num, x_cat, y) for validation).
    """
    tr, val, _ = split_by_time(df)  # test block is discarded, never named
    if n_train is not None:
        tr = tr.sample(n_train, random_state=seed)  # smaller set of TRAIN rows for fast checks; not a split
    prep = fit_prep(tr, feature_names(df))
    label = lambda s: torch.from_numpy(s["isFraud"].to_numpy("float32"))
    return prep, (*transform(tr, prep), label(tr)), (*transform(val, prep), label(val))


class Net(nn.Module):
    """Embeddings for category columns, then Linear -> ReLU -> Dropout twice, then one output (a logit)."""

    def __init__(self, n_num, slots, dropout=DROPOUT):
        super().__init__()
        self.emb = nn.ModuleList(nn.Embedding(n, min(MAX_EMB, (n + 1) // 2)) for n in slots)
        width = n_num + sum(e.embedding_dim for e in self.emb)
        layers = []
        for h in HIDDEN:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.mlp = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x_num, x_cat):
        embs = [e(x_cat[:, j]) for j, e in enumerate(self.emb)]
        return self.mlp(torch.cat([x_num, *embs], dim=1)).squeeze(1)


def slots(prep):
    return [len(prep["vocab"][c]) + 2 for c in prep["cat"]]  # + unknown + missing


def setup(seed):
    """Repeatable runs: fixed seed, CPU, fixed threads, refuse non-repeatable operations."""
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)


def run_epoch(net, opt, xn, xc, y, gen):
    """One pass over all rows in shuffled batches. Returns the average training loss."""
    net.train()  # dropout on
    loss_fn, total = nn.BCEWithLogitsLoss(), 0.0
    order = torch.randperm(len(y), generator=gen)
    for i in range(0, len(y), BATCH):
        b = order[i:i + BATCH]
        opt.zero_grad()
        loss = loss_fn(net(xn[b], xc[b]), y[b])
        loss.backward()  # backpropagation: work out which way to nudge each weight
        opt.step()       # nudge them
        total += loss.item() * len(b)
    return total / len(y)


@torch.no_grad()
def predict(net, xn, xc):
    net.eval()  # dropout off
    return torch.sigmoid(net(xn, xc)).numpy()


def scores(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}


def train(df, seed=SEED, max_epochs=MAX_EPOCHS, n_train=None, shuffle_labels=False, verbose=True):
    """Fit on train rows, early-stop on validation ROC-AUC, keep the best epoch.

    Returns (net, prep, info, train_data, val_data).
    """
    setup(seed)
    prep, (xn, xc, y), (vn, vc, vy) = prepare(df, n_train, seed)
    if shuffle_labels:  # sanity check only: scrambles which train rows are fraud
        y = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))]
    net = Net(xn.shape[1], slots(prep))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, t_start = -1.0, 0, None, time.time()
    for ep in range(1, max_epochs + 1):
        t0 = time.time()
        loss = run_epoch(net, opt, xn, xc, y, gen)
        auc = roc_auc_score(vy, predict(net, vn, vc))
        if auc > best:
            best, best_ep = auc, ep
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if verbose:
            print(f"epoch {ep:2d}/{max_epochs}  loss {loss:.4f}  val ROC-AUC {auc:.4f}  "
                  f"best {best:.4f} (ep {best_ep})  {time.time() - t0:.1f} s", flush=True)
        if ep - best_ep >= PATIENCE:
            break
    net.load_state_dict(best_state)
    info = {"best_epoch": best_ep, "epochs_run": ep, "train_seconds": round(time.time() - t_start, 1),
            "val": scores(vy.numpy(), predict(net, vn, vc))}
    return net, prep, info, (xn, xc, y), (vn, vc, vy)


def sanity(df):
    """Two checks: the code can learn (memorize 1,000 rows) and nothing leaks (shuffled labels score ~0.5)."""
    setup(SEED)
    prep, (xn, xc, y), _ = prepare(df, n_train=1000)
    net = Net(xn.shape[1], slots(prep), dropout=0.0)  # dropout off: we ask "can it learn at all?"
    opt, gen = torch.optim.Adam(net.parameters(), lr=LR), torch.Generator().manual_seed(SEED)
    for _ in range(300):
        run_epoch(net, opt, xn, xc, y, gen)
    with torch.no_grad():
        net.eval()
        loss = nn.BCEWithLogitsLoss()(net(xn, xc), y).item()
    print(f"overfit 1,000 rows ({int(y.sum())} fraud), 300 epochs: train loss {loss:.4f}  "
          f"{'PASS' if loss < 0.05 else 'FAIL'} (need < 0.05)")
    # Yardstick: networks that never saw any label. Their scores are a random function of the columns, and the
    # columns relate to fraud, so they land far from 0.5 by accident (0.29-0.68 here). A leak would beat them clearly.
    prep, (xn, _, _), (vn, vc, vy) = prepare(df)
    untrained = []
    for s in range(20):
        torch.manual_seed(s)
        untrained.append(roc_auc_score(vy, predict(Net(xn.shape[1], slots(prep)), vn, vc)))
    *_, info, _, _ = train(df, shuffle_labels=True)
    auc, limit = info["val"]["roc_auc"], max(untrained) + 0.05
    print(f"untrained nets (20 seeds): val ROC-AUC {min(untrained):.4f}-{max(untrained):.4f}")
    print(f"shuffled labels: val ROC-AUC {auc:.4f}  best epoch {info['best_epoch']}  "
          f"{'PASS' if auc <= limit else 'FAIL'} (need <= best untrained + 0.05 = {limit:.4f})")


if __name__ == "__main__":
    from src.data import load_raw

    df = load_raw()
    if "--sanity" in sys.argv:
        sanity(df)
        sys.exit()
    net, prep, info, (xn, xc, y), (vn, vc, vy) = train(df)
    sample = torch.randperm(len(y), generator=torch.Generator().manual_seed(SEED))[:TRAIN_SAMPLE]
    tr_s = scores(y[sample].numpy(), predict(net, xn[sample], xc[sample]))
    n_params = sum(p.numel() for p in net.parameters())
    out = {
        "seed": SEED, "n_features": len(prep["num"]) + len(prep["cat"]),
        "n_numeric": len(prep["num"]), "n_category": len(prep["cat"]), "n_missing_flags": len(prep["flag"]),
        "n_inputs": net.mlp[0].in_features, "n_params": n_params, "n_train": len(y), "n_val": len(vy),
        "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"], "train_seconds": info["train_seconds"],
        "config": {"hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "batch": BATCH, "max_epochs": MAX_EPOCHS,
                   "patience_epochs": PATIENCE, "min_count": MIN_COUNT, "clip": CLIP, "threads": THREADS,
                   "device": "cpu"},
        "train_sample": {"rows": TRAIN_SAMPLE, **tr_s}, "model": info["val"], "no_skill": no_skill(vy.numpy()),
    }
    (ROOT / "metrics").mkdir(exist_ok=True)
    (ROOT / "metrics" / "phase3_nn.json").write_text(json.dumps(out, indent=2))

    lgbm = json.loads((ROOT / "metrics" / "phase2_lgbm.json").read_text())
    m, g = out["model"], lgbm["model"]
    print(f"\n{'':24}{'neural net':>14}{'LightGBM':>14}")
    print(f"{'val ROC-AUC':24}{m['roc_auc']:14.4f}{g['roc_auc']:14.4f}")
    print(f"{'val PR-AUC':24}{m['pr_auc']:14.4f}{g['pr_auc']:14.4f}")
    print(f"{'train-sample ROC-AUC':24}{tr_s['roc_auc']:14.4f}{'n/a':>14}  (not recorded in Phase 2)")
    print(f"{'train-sample PR-AUC':24}{tr_s['pr_auc']:14.4f}{'n/a':>14}")
    print(f"{'training time (s)':24}{out['train_seconds']:14.0f}{'~55':>14}  (Phase 2 notes)")
    print(f"{'size':24}{n_params:>8,} wts{lgbm['best_iteration']:>8} trees")
    print(f"{'no-skill ROC / PR':24}{'0.5000 / ' + format(out['no_skill']['pr_auc'], '.4f'):>14}")
    print("\nONE SEED: this cannot say which model is better. Seed-to-seed spread is measured in Phase 4.")
