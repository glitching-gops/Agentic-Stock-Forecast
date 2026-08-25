"""
tools/lora_kaggle.py - LoRA fine-tuning of Chronos-2, to run in a Kaggle GPU notebook.

HOW TO RUN
----------
1. `python tools/export_lora_package.py --out lora_package.npz` locally.
2. Upload that one file (about 2 MB) as a Kaggle dataset.
3. New Kaggle notebook, accelerator GPU (P100 or T4), then:

       !pip -q install chronos-forecasting peft
       !cp /kaggle/input/<your-dataset>/lora_package.npz .
       # paste this file, or add it to the dataset and %run it
       PACKAGE = "lora_package.npz"

4. Download `lora_predictions.npz` and score it locally:

       python tools/score_lora.py --predictions lora_predictions.npz

WHAT IS AND IS NOT DECIDED HERE
-------------------------------
Nothing methodological. The purged folds, the as-of position of every window,
the target and the train/test membership were all decided by the same tested
code that produced every other row of the results table, and arrive as data.
This file does gradient descent and nothing else.

That matters more than it sounds. The single most dangerous thing about moving
an experiment to a notebook is that the notebook re-derives a boundary slightly
differently and produces a number that looks fine and is not comparable with
anything. `end_index` is the causal decision - an off-by-one there hands the
model its own answer and reads as a breakthrough - and it is computed locally,
not here. The only thing this file does with it is slice.

THE EXPERIMENT
--------------
A linear probe on the FROZEN encoder state scored reb_t +0.87 at context 512
and -0.13 at 2048, against a pre-registered bar of 2. That says the frozen
representation does not linearly encode the 30-session excess return.

LoRA asks the remaining question: can the representation be ADAPTED into one
that does? Everything else is held identical to the probe - the same encoder
state, the same linear head, the same MSE objective, the same folds and rows -
so the only difference is that the encoder is now allowed to move. If this also
comes back null, the two results together say the signal is not there to be
found, rather than that we used the wrong read-out.
"""

import json
import time

import numpy as np
import torch
import torch.nn as nn

import os

# Read from the environment so a local smoke run can shrink the job without
# editing the file that gets pasted into Kaggle - the pasted file and the
# tested file must be the same bytes.
PACKAGE = os.environ.get("LORA_PACKAGE", "lora_package.npz")
OUT = os.environ.get("LORA_OUT", "lora_predictions.npz")
LIMIT_FOLDS = int(os.environ.get("LORA_LIMIT_FOLDS", "0"))   # 0 = all
LIMIT_TRAIN = int(os.environ.get("LORA_LIMIT_TRAIN", "0"))   # 0 = all
LIMIT_TEST = int(os.environ.get("LORA_LIMIT_TEST", "0"))     # 0 = all

MODEL_ID = "amazon/chronos-2"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ["q", "v"]          # self_attention projections; see below
EPOCHS = int(os.environ.get("LORA_EPOCHS", "3"))
BATCH = int(os.environ.get("LORA_BATCH", "32"))
EVAL_BATCH = int(os.environ.get("LORA_EVAL_BATCH", "128"))
LR_LORA = float(os.environ.get("LORA_LR", "1e-4"))
LR_HEAD = float(os.environ.get("LORA_LR_HEAD", "1e-3"))
SEED = 17


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── the data ────────────────────────────────────────────────────────────────

pkg = np.load(PACKAGE, allow_pickle=True)
META = json.loads(str(pkg["meta"][0]))
SERIES = pkg["series"]                       # (n_tickers, n_dates) float32
ROW_FOLD = pkg["row_fold"]
ROW_SPLIT = pkg["row_split"]                 # 0 train, 1 test
ROW_TICKER = pkg["row_ticker"]
ROW_END = pkg["row_end"]
ROW_TARGET = pkg["row_target"]
CONTEXT = int(META["context"])
HORIZON = int(META["horizon"])
N_FOLDS = int(META["folds"])

log(f"package: {len(ROW_FOLD):,} rows | context {CONTEXT} | horizon {HORIZON} "
    f"| {META['n_tickers']} tickers | stride {META['train_stride']}")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
# TF32 keeps the exponent but cuts the float32 mantissa from 23 bits to 10.
# The local CPU/GPU agreement that validates every measured table in this
# project was 4.8e-07, and it only holds with TF32 off.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def windows(idx: np.ndarray) -> np.ndarray:
    """Left-padded context windows for these rows. NaN marks 'no observation'."""
    out = np.full((len(idx), CONTEXT), np.nan, dtype=np.float32)
    for k, i in enumerate(idx):
        end = int(ROW_END[i])
        start = max(0, end - CONTEXT + 1)
        chunk = SERIES[int(ROW_TICKER[i]), start:end + 1]
        out[k, CONTEXT - len(chunk):] = chunk
    return out


# ── the model ───────────────────────────────────────────────────────────────

def build():
    from chronos import Chronos2Pipeline
    from peft import LoraConfig, get_peft_model

    pipe = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)
    model = pipe.model
    for p in model.parameters():
        p.requires_grad = False

    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                     target_modules=LORA_TARGETS, lora_dropout=0.05,
                     bias="none")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  LoRA r={LORA_RANK} on {LORA_TARGETS}: {trainable:,} trainable params")
    return pipe, model


N_OUT_PATCHES = 2                  # 30-session horizon at a 16-session patch
D_MODEL = 768
EMBED_DIM = N_OUT_PATCHES * D_MODEL


def encode(model, ctx: torch.Tensor) -> torch.Tensor:
    """
    The encoder state the forecast head reads, differentiably.

    The probe took this with a forward hook under `no_grad`, driven through
    `pipe.predict`. Training needs gradients, so the model is called directly -
    and the argument shapes were read off a real `predict` call rather than
    guessed: context is raw (batch, time), group_ids is (batch,), and
    future_covariates is (batch, output_patches * patch_size) of NaN when there
    are no covariates, which is our case.

    group_ids is arange, i.e. every series independent. Collapsing it to zeros
    is Chronos-2's cross-learning, which measured consistently NEGATIVE on this
    panel at both contexts.
    """
    b = ctx.shape[0]
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    enc_out, _, _, _ = base.encode(
        context=ctx,
        group_ids=torch.arange(b, device=ctx.device),
        future_covariates=torch.full((b, N_OUT_PATCHES * 16), float("nan"),
                                     device=ctx.device, dtype=ctx.dtype),
        num_output_patches=N_OUT_PATCHES,
    )
    h = enc_out.last_hidden_state
    return h[:, -N_OUT_PATCHES:, :].reshape(b, -1)


class Head(nn.Module):
    """LayerNorm then linear. The same read-out the frozen probe used."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.fc = nn.Linear(EMBED_DIM, 1)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.fc.weight, std=0.01)

    def forward(self, z):
        return self.fc(self.norm(z)).squeeze(-1)


def self_check(pipe, model):
    """
    The hand-built forward must match the pipeline's own path.

    This is the one thing in this file that could silently differ from the
    locally-scored comparators: if `encode` is called with arguments the
    pipeline would not have used, every number downstream describes a different
    computation. So the same batch is pushed through both and the encoder
    states compared.
    """
    rng = np.random.default_rng(0)
    hist = [np.cumsum(rng.normal(0, 0.01, 400)).astype(np.float32) - 3.0
            for _ in range(4)]

    captured = []
    h = pipe.model.encoder.register_forward_hook(
        lambda _m, _i, o: captured.append(
            (o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
             ).detach()))
    try:
        with torch.no_grad():
            pipe.predict(hist, prediction_length=HORIZON, batch_size=4)
    finally:
        h.remove()

    ctx = torch.full((4, CONTEXT), float("nan"), dtype=torch.float32,
                     device=device)
    for i, s in enumerate(hist):
        ctx[i, CONTEXT - len(s):] = torch.tensor(s, device=device)

    with torch.no_grad():
        mine = encode(model, ctx)
    theirs = captured[0][:, -N_OUT_PATCHES:, :].reshape(4, -1)

    diff = float((mine - theirs).abs().max())
    log(f"  self-check: max |hand-built - pipeline| = {diff:.3e}")
    if diff > 1e-3:
        raise RuntimeError(
            f"the hand-built forward disagrees with the pipeline by {diff:.3e}. "
            f"Every number this notebook produces would describe a different "
            f"computation from the locally-scored comparators.")


# -- train and predict, one fold at a time -----------------------------------

def main():
    all_rows, all_pred = [], []
    folds = LIMIT_FOLDS or N_FOLDS
    only = os.environ.get("LORA_ONLY_FOLD")

    for fold in ([int(only)] if only else range(folds)):
        log(f"fold {fold}")
        tr = np.flatnonzero((ROW_FOLD == fold) & (ROW_SPLIT == 0))
        te = np.flatnonzero((ROW_FOLD == fold) & (ROW_SPLIT == 1))
        if LIMIT_TRAIN:
            # TAIL takes the training rows nearest the test window, which is
            # what makes a small-data fold comparable to fold 0 rather than
            # merely smaller and older.
            tr = (tr[-LIMIT_TRAIN:] if os.environ.get("LORA_TRAIN_TAIL")
                  else tr[:LIMIT_TRAIN])
        if LIMIT_TEST:
            te = te[:LIMIT_TEST]
        log(f"  {len(tr):,} train rows, {len(te):,} test rows")

        pipe, model = build()
        if fold == 0:
            self_check(pipe, model)

        head_net = Head().to(device)
        opt = torch.optim.AdamW(
            [{"params": [q for q in model.parameters() if q.requires_grad],
              "lr": LR_LORA},
             {"params": head_net.parameters(), "lr": LR_HEAD}],
            weight_decay=0.01)

        # Target standardised on TRAINING rows only, undone at predict time.
        y_mu = float(ROW_TARGET[tr].mean())
        y_sd = float(ROW_TARGET[tr].std()) or 1.0

        rng = np.random.default_rng(SEED + fold)
        model.train()
        for epoch in range(EPOCHS):
            order = rng.permutation(len(tr))
            running, seen = 0.0, 0
            for start in range(0, len(order), BATCH):
                idx = tr[order[start:start + BATCH]]
                ctx = torch.tensor(windows(idx), device=device)
                y = torch.tensor((ROW_TARGET[idx] - y_mu) / y_sd, device=device)

                pred = head_net(encode(model, ctx))
                loss = nn.functional.mse_loss(pred, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [q for q in model.parameters() if q.requires_grad]
                    + list(head_net.parameters()), 1.0)
                opt.step()

                running += float(loss.detach()) * len(idx)
                seen += len(idx)
            log(f"  epoch {epoch}: train MSE {running / max(seen, 1):.5f}")

        model.eval()
        head_net.eval()
        preds = np.zeros(len(te), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(te), EVAL_BATCH):
                idx = te[start:start + EVAL_BATCH]
                ctx = torch.tensor(windows(idx), device=device)
                preds[start:start + len(idx)] = (
                    head_net(encode(model, ctx)).float().cpu().numpy()
                    * y_sd + y_mu)

        all_rows.append(te)
        all_pred.append(preds)
        log(f"  predicted {len(te):,} rows (pred sd {preds.std():.5f} vs "
            f"target sd {ROW_TARGET[te].std():.5f})")

        del model, pipe, head_net, opt
        torch.cuda.empty_cache()

    rows = np.concatenate(all_rows)
    np.savez_compressed(
        OUT,
        row_index=rows.astype(np.int32),
        prediction=np.concatenate(all_pred).astype(np.float32),
        config=np.array([json.dumps({
            "model_id": MODEL_ID, "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA, "targets": LORA_TARGETS,
            "epochs": EPOCHS, "batch": BATCH, "lr_lora": LR_LORA,
            "lr_head": LR_HEAD, "context": CONTEXT, "seed": SEED,
            "folds_run": folds,
        })], dtype=object),
    )
    log(f"wrote {OUT}: {len(rows):,} predictions across {folds} fold(s)")
    log("Score it locally: python tools/score_lora.py --predictions " + OUT)


# A pasted Kaggle cell has __name__ == "__main__", so this still runs
# top-to-bottom there; importing the file for a local smoke test does not.
if __name__ == "__main__":
    main()
