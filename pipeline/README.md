# Pipeline: raw archive → entity table → task_c

These scripts turn the ITS JPO raw archive (`raw.tar`, `processed.tar`, ~200 GB) into the parquet files used by `bench/`.
They are kept as they were run; paths are hard-coded to `C:/wzdx/`.

Expected order (copy the originals from `C:/wzdx/` into this folder):

1. `peek_tar.py`, `peek_tar2.py` — inspect archive layout
2. `readfile.py`, `read_wz.py`, `kinds.py` — parse one snapshot / enumerate record kinds
3. `extract.py` → `extract_full.py` — collapse snapshots by entity ID into `out/north_carolina/entities_*.parquet`
4. `label_slip.py` — select entities whose posted end changed, apply filters R1–R4, write `out/task_c_slip_nc.parquet`

`label_slip.py` is the provenance of every label in the benchmark; see paper §2.
