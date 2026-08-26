# Vendored: Kronos

Source: https://github.com/shiyu-coder/Kronos
Commit: `67b630e67f6a18c9e9be918d9b4337c960db1e9a` (2026-04-13)
Files:  `model/__init__.py`, `model/kronos.py`, `model/module.py`, `LICENSE`
Licence: MIT (Copyright (c) 2025 ShiYu) — retained verbatim beside the code.

## Why vendored rather than installed

Kronos ships **no pip package**. The published usage is
`from model import Kronos, KronosTokenizer, KronosPredictor` against a git
clone, which in CI is an unpinned dependency whose contents can change under a
recorded result — the same reason `requirements-series.txt` refuses the
`timesfm` PyPI package and reaches TimesFM-2.5 through `transformers` instead.

Three files, ~54 KB, MIT. Pinning the commit here makes the model code part of
what `config_hash` describes, so a run before and after an upstream change is
distinguishable in `experiment_runs`.

## Modifications

Exactly one, marked inline with `# VENDOR PATCH`:

- `kronos.py:10` — `from model.module import *` → `from .module import *`.
  The upstream absolute import assumes a top-level `model` package. Left as-is
  it would either fail to import or, worse, silently bind to some other
  `model` module on `sys.path` — and this repo has `pipeline/model.py`.

Nothing else is edited. Re-pin by re-downloading all four files at a new SHA
and re-applying that one line; do not hand-merge.

## Audited on download

No `exec`, `eval`, `subprocess`, `os.system`, `socket`, `urllib`, `requests`
or `pickle.load` anywhere in the three files. Imports are torch, numpy, pandas,
einops, huggingface_hub, tqdm, math, sys.
