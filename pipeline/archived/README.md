# Archived models

These modules are **not** on the live path. They were removed in Phase 0.

## `lstm_model.py`

Never produced a forecast in production. `df_full['target']` carries `NaN` for
the trailing 30 rows, those rows land in the 20% validation tail, so `val_loss`
is `NaN`. The guard `if val_loss < best_val_loss` is never true for `NaN`, so
`torch.save()` was never reached; training early-stopped at epoch 10 and the
subsequent `torch.load(model_path)` raised `FileNotFoundError`, which the caller
swallowed. `lstm_price` was therefore always `None`.

Two further defects would need fixing before it could be trusted:

- The `StandardScaler` is fitted on the full frame *before* the train/val split,
  so validation statistics leak into training.
- Features are standardised but the target is a raw price, leaving a badly
  conditioned regression against an unscaled output.

## `meta_learner.py`

The Ridge meta-learner was fitted on the validation set and then scored on that
same validation set. That is the origin of the project's reported ~4.3% MAPE and
~85% directional accuracy; re-running the procedure on live NSE data reproduces
those figures from pure in-sample fit.

## Reinstatement

Either module may return, but only through an experiment that earns it —
experiment E3 in the audit's plan compares a non-linear model against a linear
factor baseline on identical features under purged walk-forward evaluation. If
the ensemble does not beat that baseline out of sample, it stays here.
