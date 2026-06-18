#!/usr/bin/env python
"""Grain ArrayRecord reader proof on the sample.

Proves:
  (1) grain.IndexSampler(shuffle=True) over ArrayRecordDataSource gives a TRUE
      global, cross-file/cross-dataset permutation of all 410 records.
  (2) A batch assembled from the ArrayRecord path (decode -> collate_embeds_pack)
      is IDENTICAL to the same rows read straight from parquet and collated the
      same way (byte-exact batch == parquet path).
"""
import json, os
import numpy as np
import ml_dtypes
import gcsfs, msgpack
import pyarrow.parquet as pq
import grain.python as gp
from easydel.data.transforms.collators import collate_embeds_pack

OUT = "/tmp/ar_sample"
fs = gcsfs.GCSFileSystem(token="google_default")
man = json.load(open(os.path.join(OUT, "_manifest.json")))
records = man["records"]                       # write-order == source-index order
files = sorted(f for f in os.listdir(OUT) if f.endswith(".array_record"))

# map basename(part) -> full gs path (for parquet re-read in proof #2)
PART_GS = {
 "part-14-00000-r00000.parquet": "uscentral2stuff/data/vlm_v2/source=FineVision__SynthChartNet/area_bucket=1024/slen_band=1088/part-14-00000-r00000.parquet",
 "part-aguvis-s0-00000-r00000.parquet": "uscentral2stuff/data/vlm_v2/source=AGUVIS/area_bucket=1024/slen_band=1088/part-aguvis-s0-00000-r00000.parquet",
 "part-aria6-s0-00000-r00000.parquet": "uscentral2stuff/data/vlm_v2/source=Aria-UI/area_bucket=1536/slen_band=1088/part-aria6-s0-00000-r00000.parquet",
 "part-15-00000-r00000.parquet": "uscentral2stuff/data/vlm_v2/source=FineVision__SynthFormulaNet/area_bucket=512/slen_band=1088/part-15-00000-r00000.parquet",
}

# ArrayRecordDataSource concatenates files in the order given -> global index ranges.
paths = [os.path.join(OUT, f) for f in files]
src = gp.ArrayRecordDataSource(paths)
N = len(src)
# global_index -> manifest entry. ArrayRecordDataSource(paths) concatenates files in
# `paths` order (== `files`, alphabetical); the manifest was written in PARTS order, so
# rebuild the global mapping by walking files in paths order, each file's rows by local_idx.
by_file = {}
for m in records:
    by_file.setdefault(m["file"], []).append(m)
for v in by_file.values():
    v.sort(key=lambda m: m["local_idx"])
gidx_meta = []
for f in files:
    gidx_meta.extend(by_file[f])
assert len(gidx_meta) == N, (len(gidx_meta), N)

def gi_label(gi):
    m = gidx_meta[gi]
    return m["file"].split("__")[0], m["subset"]

# ---- PROOF 1: global cross-file/cross-dataset shuffle ----
samp = gp.IndexSampler(num_records=N, shard_options=gp.ShardOptions(0, 1, drop_remainder=True),
                       shuffle=True, seed=42, num_epochs=1)
keys = [md.record_key for md in samp]
print(f"=== PROOF 1: global shuffle ===  epoch records: {len(keys)}")
print("is permutation of [0,N):", sorted(keys) == list(range(N)))
print("first 24 shuffled (global_idx -> dataset):")
for gi in keys[:24]:
    src_name, subset = gi_label(gi)
    print(f"   {gi:4d} -> {src_name}/{subset}")
# distinct datasets seen in the first 16 draws (interleaving across files)
firstN = [gi_label(gi)[0] for gi in keys[:16]]
print("distinct source-files in first 16 draws:", sorted(set(firstN)), f"({len(set(firstN))}/4)")

# ---- PROOF 2: batch from ArrayRecord == batch from parquet ----
print("\n=== PROOF 2: batch matches parquet path ===")
B = 6
batch_keys = keys[:B]

def decode_ar(gi):
    return msgpack.unpackb(src[gi], raw=False)

def read_parquet_row(part_basename, row_idx):
    pf = pq.ParquetFile(fs.open(PART_GS[part_basename], "rb"))
    seen = 0
    for rg in range(pf.num_row_groups):
        nrg = pf.metadata.row_group(rg).num_rows
        if row_idx < seen + nrg:
            cols = pf.read_row_group(rg).to_pydict()
            j = row_idx - seen
            return {k: cols[k][j] for k in cols}
        seen += nrg
    raise IndexError(row_idx)

rows_ar = [decode_ar(gi) for gi in batch_keys]
rows_pq = [read_parquet_row(gidx_meta[gi]["src_part"], gidx_meta[gi]["src_row"]) for gi in batch_keys]

# detect image_token_id from data: the id repeated == sum(embed_n_tok)
def detect_img_tok(row):
    ent = sum(int(x) for x in row["embed_n_tok"])
    if ent <= 0:
        return None
    from collections import Counter
    c = Counter(row["input_ids"])
    for tok, cnt in c.most_common():
        if cnt == ent:
            return int(tok)
    return None

img_tok = next((detect_img_tok(r) for r in rows_ar if detect_img_tok(r) is not None), None)
embed_dim = int(rows_ar[0]["embed_dim"])
# max_total = embed-row CAPACITY = total image tokens across the batch (not seq len)
max_total = sum(int(x) for r in rows_ar for x in r["embed_n_tok"])
print(f"image_token_id={img_tok}  embed_dim={embed_dim}  max_total(embed_rows)={max_total}  B={B}")

kw = dict(pad_id=0, max_total=max_total, image_token_id=img_tok,
          embed_dim=embed_dim, embed_dtype=ml_dtypes.bfloat16)
b_ar = collate_embeds_pack(rows_ar, **kw)
b_pq = collate_embeds_pack(rows_pq, **kw)

print("batch keys:", sorted(b_ar.keys()))
all_ok = b_ar.keys() == b_pq.keys()
for k in b_ar:
    a, p = np.asarray(b_ar[k]), np.asarray(b_pq[k])
    eq = a.shape == p.shape and np.array_equal(a, p)
    all_ok &= eq
    print(f"   {k:20s} shape={str(a.shape):18s} dtype={str(a.dtype):16s} equal={eq}")
print("RESULT:", "BATCH MATCHES PARQUET PATH (byte-exact)" if all_ok else "FAIL")
