#!/usr/bin/env python
"""Verify the EXACT weighted-mix production pipeline (MapDataset API) + throughput,
so the trainer wiring uses a confirmed API. per-ds shuffle -> mix(weights) -> shard
-> decode -> batch(batch_fn) -> to_iter_dataset(prefetch)."""
import inspect, time
import numpy as np, ml_dtypes, msgpack
import grain.python as gp
from easydel.data.transforms.collators import collate_embeds_pack

print("to_iter_dataset sig:", inspect.signature(gp.MapDataset.to_iter_dataset))
print("slice sig:", inspect.signature(gp.MapDataset.slice))

BASE = "gs://uscentral2stuff/tmp/ar_arrecord_sample"
DSF = {
 "AGUVIS":   f"{BASE}/AGUVIS__b1024__s1088__part-aguvis-s0-00000-r00000.array_record",
 "Aria-UI":  f"{BASE}/Aria-UI__b1536__s1088__part-aria6-s0-00000-r00000.array_record",
 "ChartNet": f"{BASE}/FineVision__SynthChartNet__b1024__s1088__part-14-00000-r00000.array_record",
 "FormNet":  f"{BASE}/FineVision__SynthFormulaNet__b512__s1088__part-15-00000-r00000.array_record",
}
IMG_TOK = 248056
names = sorted(DSF)
weights = [0.4, 0.3, 0.2, 0.1]

class MsgpackDecode(gp.MapTransform):
    def map(self, b): return msgpack.unpackb(b, raw=False)

def collate(rows):
    mt = sum(int(x) for r in rows for x in r["embed_n_tok"])
    return collate_embeds_pack(rows, pad_id=0, max_total=mt, image_token_id=IMG_TOK,
                               embed_dim=5120, embed_dtype=ml_dtypes.bfloat16)

def build(shard_index, shard_count, B, num_epochs, num_threads=16, prefetch=64, do_shuffle=True):
    per = []
    for i, nm in enumerate(names):
        ds = gp.MapDataset.source(gp.ArrayRecordDataSource([DSF[nm]]))
        if do_shuffle: ds = ds.shuffle(seed=42 + i)
        per.append(ds)
    mixed = gp.MapDataset.mix(per, weights=weights)
    if num_epochs > 1: mixed = mixed.repeat(num_epochs)
    if shard_count > 1: mixed = mixed.slice(slice(shard_index, None, shard_count))
    decoded = mixed.map(MsgpackDecode())
    batched = decoded.batch(batch_size=B, drop_remainder=True, batch_fn=collate)
    it = batched.to_iter_dataset(read_options=gp.ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch))
    return mixed, batched, it

# 1) len + shard disjointness (use index tags, no decode)
print("\n=== shard disjointness (stride) ===")
tag_per = [gp.MapDataset.source(gp.ArrayRecordDataSource([DSF[nm]])).map(lambda b, i=gi: i)
           for gi, nm in enumerate(names)]
gmix = gp.MapDataset.mix(tag_per, weights=weights)
try:
    print("mixed len:", len(gmix))
except Exception as e:
    print("mixed len err:", e)
s0 = [gmix.slice(slice(0, None, 2))[i] for i in range(10)]
s1 = [gmix.slice(slice(1, None, 2))[i] for i in range(10)]
print("shard0[:10] tags:", s0)
print("shard1[:10] tags:", s1)

# 2) full pipeline: pull batches + throughput
print("\n=== full pipeline (B=8, shard 0/1, epochs=4) ===")
mixed, batched, it = build(0, 1, B=8, num_epochs=4)
try: print("batched steps (len):", len(batched))
except Exception as e: print("batched len err:", e)
i = iter(it)
t0 = time.time(); first = next(i); tf = time.time() - t0
print("first batch keys:", sorted(first.keys()), "img_embeds shape:", np.asarray(first["image_embeds"]).shape)
ts = time.time(); nb = 20
for _ in range(nb): b = next(i)
dt = time.time() - ts
print(f"first={tf:.2f}s  steady {dt/nb*1000:.1f} ms/batch  {nb*8/dt:.1f} rows/s  (B=8)")
print("PIPELINE OK")
