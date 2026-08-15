# Scripts

Utility and migration scripts. These are run manually as needed, not as part of the main application.

## migrate_to_supabase.py
One-time migration script to transfer data from local SQLite to Supabase PostgreSQL.
Run once during initial deployment setup.

## data_check.py
Utility script to check data sanity in the database.

## verify_endpoints.py
Utility script to verify the health of the FastAPI endpoints.

## verify_stage1.py & verify_stage2.py
Validation scripts used during initial pipeline execution and universe scaling to verify system outputs.

## Phase 0 changes

- `select_top_50.py` — **deleted.** It ranked stocks by composite score and kept
  the top 5 per sector, which selected the universe on the model's own reported
  accuracy (audit finding F4). The universe now comes from `data/universe.py`,
  which applies a point-in-time rule referencing no model output.
- `update_tickers.py` — **deleted.** It rewrote the hard-coded `TICKERS` dict in
  `data/tickers.py`. That dict no longer defines the universe; `tickers.py` now
  holds metadata only and reads it from the `index_membership` table.
- The remaining tools take their ticker list from `data.universe.get_universe()`.

### Recovering historical index membership

`data/universe.py` records membership from the first `sync_current_membership()`
call onward, so history before that date is unknown and evaluations covering it
are survivorship-biased. `backfill_membership_from_wayback()` reconstructs
earlier membership from Internet Archive snapshots of the NSE constituent CSV.
It is a manual tool because archive.org's CDX endpoint is frequently
unavailable — it returns 0 and reports the failure rather than writing partial
history that could be mistaken for complete. Re-run it periodically:

```bash
python -c "from data.universe import backfill_membership_from_wayback as b; print(b())"
```
