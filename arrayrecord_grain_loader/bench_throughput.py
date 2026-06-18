#!/usr/bin/env python
"""Data-wait benchmark: can the grain ArrayRecord loader sustain >= the step rate?

Reads .array_record straight from gs:// under TRUE global shuffle (worst-case random
access), full production op chain (msgpack-decode -> per-batch pack collate), with
grain parallel prefetch. Reports steady-state per-batch produce time + rows/s + MB/s,
to compare against the ~22 s/step budget. data-wait per step ~= max(0, T_batch - T_step).
"""
import time, sys
import numpy as np, ml_dtypes, msgpack
import grain.python as gp
from easydel.data.transforms.collators import collate_embeds_pack

BASE = "gs://uscentral2stuff/tmp/ar_arrecord_sample"
FILES = [f"{BASE}/{n}" for n in [
    "AGUVIS__b1024__s1088__part-aguvis-s0-00000-r00000.array_record",
    "Aria-UI__b1536__s1088__part-aria6-s0-00000-r00000.array_record",
    "FineVision__SynthChartNet__b1024__s1088__part-14-00000-r00000.array_record",
    "FineVision__SynthFormulaNet__b512__s1088__part-15-00000-r00000.array_record",
]]
IMG_TOK = 248056
STEP_BUDGET_S = 22.0

class MsgpackDecode(gp.MapTransform):
    def map(self, b):
        return msgpack.unpackb(b, raw=False)

def make_collate(stats):
    def collate(rows):
        max_total = sum(int(x) for r in rows for x in r["embed_n_tok"])
        stats["embed_bytes"] += sum(len(bb) for r in rows for bb in (r.get("image_embeds") or []))
        stats["rows"] += len(rows)
        return collate_embeds_pack(rows, pad_id=0, max_total=max_total,
                                   image_token_id=IMG_TOK, embed_dim=5120,
                                   embed_dtype=ml_dtypes.bfloat16)
    return collate

def bench(B, num_threads, prefetch, n_batches=30, worker_count=0):
    src = gp.ArrayRecordDataSource(FILES)
    N = len(src)
    samp = gp.IndexSampler(num_records=N, shard_options=gp.ShardOptions(0, 1, drop_remainder=True),
                           shuffle=True, seed=7, num_epochs=50)
    stats = {"embed_bytes": 0, "rows": 0}
    dl = gp.DataLoader(
        data_source=src, sampler=samp,
        operations=[MsgpackDecode(), gp.Batch(B, drop_remainder=True, batch_fn=make_collate(stats))],
        worker_count=worker_count, worker_buffer_size=2,
        read_options=gp.ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch),
    )
    it = iter(dl)
    t0 = time.time(); first = next(it); t_first = time.time() - t0   # incl. startup/prefetch warmup
    # steady state
    ts = time.time()
    s_rows0, s_bytes0 = stats["rows"], stats["embed_bytes"]
    for _ in range(n_batches):
        _ = next(it)
    dt = time.time() - ts
    rows = stats["rows"] - s_rows0
    mb = (stats["embed_bytes"] - s_bytes0) / 1e6
    per_batch = dt / n_batches
    print(f"B={B:3d} thr={num_threads:2d} pf={prefetch:3d} wc={worker_count} | "
          f"first={t_first:5.2f}s | steady {per_batch*1000:7.1f} ms/batch  "
          f"{rows/dt:6.1f} rows/s  {mb/dt:7.1f} MB/s | "
          f"data-wait/step@22s={max(0.0, per_batch-STEP_BUDGET_S):.2f}s  "
          f"headroom={STEP_BUDGET_S/per_batch:5.1f}x")
    return per_batch

print("=== grain ArrayRecord loader throughput from gs:// (global shuffle) ===")
for B in (8, 16, 32):
    bench(B, num_threads=16, prefetch=64)
print("--- thread sweep at B=16 ---")
for thr in (4, 8, 32):
    bench(16, num_threads=thr, prefetch=64)
