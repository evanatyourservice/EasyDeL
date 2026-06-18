#!/usr/bin/env python
"""Completeness proofs: (A+B) MULTI-IMAGE round-trip + multi-span collate scatter;
(C) WEIGHTED grain mixing on real ArrayRecord (controllable weights, not size-proportional)."""
import hashlib, os
import numpy as np, ml_dtypes, msgpack
import gcsfs, pyarrow.parquet as pq
import grain.python as gp
from array_record.python.array_record_module import ArrayRecordWriter, ArrayRecordReader
from easydel.data.transforms.collators import collate_embeds_pack

fs = gcsfs.GCSFileSystem(token="google_default")
IMG_TOK = 248056
OUT = "/tmp/ar_multi"; os.makedirs(OUT, exist_ok=True)
MIMIC = "uscentral2stuff/data/vlm_v2/source=FineVision__mimic_cgd/area_bucket=1024/slen_band=192/part-12-00000-r00000.parquet"

def per_image_sha(image_embeds):
    return [hashlib.sha256(b).hexdigest() for b in (image_embeds or [])]

# ---------- A: MULTI-IMAGE convert + byte-exact (per-image) ----------
print("=== A: multi-image convert + byte-exact verify ===")
pf = pq.ParquetFile(fs.open(MIMIC, "rb"))
arpath = os.path.join(OUT, "mimic.array_record")
w = ArrayRecordWriter(arpath, "group_size:1")
src_meta = []
n = 0
for rg in range(pf.num_row_groups):
    cols = pf.read_row_group(rg).to_pydict()
    for i in range(len(cols["n_images"])):
        row = {k: cols[k][i] for k in cols}
        w.write(msgpack.packb(row, use_bin_type=True))
        src_meta.append({
            "n_images": int(row["n_images"]),
            "embed_n_tok": [int(x) for x in row["embed_n_tok"]],
            "per_img_sha": per_image_sha(row["image_embeds"]),
            "embed_dim": int(row["embed_dim"]),
        })
        n += 1
        if n >= 24: break
    if n >= 24: break
w.close()
r = ArrayRecordReader(arpath); recs = r.read_all()[:n]; r.close()
mism = 0; multi_rows = 0
for j, m in enumerate(src_meta):
    row = msgpack.unpackb(recs[j], raw=False)
    if int(row["n_images"]) >= 2: multi_rows += 1
    # n_images == len(image_embeds) == len(embed_n_tok), per-image SHA byte-exact
    ok = (int(row["n_images"]) == len(row["image_embeds"]) == len(row["embed_n_tok"]) == m["n_images"]
          and per_image_sha(row["image_embeds"]) == m["per_img_sha"])
    if not ok: mism += 1
print(f"rows={n}  multi-image rows(n_images>=2)={multi_rows}  per-image-SHA mismatches={mism}")
print("A RESULT:", "BYTE-EXACT MULTI-IMAGE PASS" if mism == 0 and multi_rows > 0 else "FAIL")

# ---------- B: multi-span collate scatter (ArrayRecord vs parquet) ----------
print("\n=== B: multi-image collate (multi-span scatter) ===")
B = 4
rows_ar = [msgpack.unpackb(recs[j], raw=False) for j in range(B)]
def read_pq_row(idx):
    seen = 0
    for rg in range(pf.num_row_groups):
        k = pf.metadata.row_group(rg).num_rows
        if idx < seen + k:
            cols = pf.read_row_group(rg).to_pydict(); return {c: cols[c][idx-seen] for c in cols}
        seen += k
rows_pq = [read_pq_row(j) for j in range(B)]
embed_dim = int(rows_ar[0]["embed_dim"])
max_total = sum(int(x) for r in rows_ar for x in r["embed_n_tok"])
n_images_per_row = [int(r["n_images"]) for r in rows_ar]
spans_per_row = [sum(1 for t in np.asarray(r["input_ids"]) if t == IMG_TOK) for r in rows_ar]  # placeholder count
embedtok_per_row = [sum(int(x) for x in r["embed_n_tok"]) for r in rows_ar]
print(f"n_images/row={n_images_per_row}  placeholder_tok/row={spans_per_row}  embed_tok/row={embedtok_per_row}")
kw = dict(pad_id=0, max_total=max_total, image_token_id=IMG_TOK, embed_dim=embed_dim, embed_dtype=ml_dtypes.bfloat16)
b_ar = collate_embeds_pack(rows_ar, **kw)
b_pq = collate_embeds_pack(rows_pq, **kw)
all_ok = b_ar.keys() == b_pq.keys()
for k in b_ar:
    a, p = np.asarray(b_ar[k]), np.asarray(b_pq[k])
    all_ok &= (a.shape == p.shape and np.array_equal(a, p))
# multi-span check: each multi-image row contributes embeds for BOTH images; positions cover all placeholders
pos = b_ar["image_embed_positions"]; mask = b_ar["image_embed_mask"]
n_scattered = int(mask.sum())
print(f"scattered embed rows={n_scattered}  expected(sum embed_tok)={sum(embedtok_per_row)}  "
      f"image_embeds shape={np.asarray(b_ar['image_embeds']).shape}")
multi_ok = (n_scattered == sum(embedtok_per_row) == sum(spans_per_row)) and all(ni >= 1 for ni in n_images_per_row) and max(n_images_per_row) >= 2
print("B RESULT:", "MULTI-SPAN BATCH MATCHES PARQUET (byte-exact) + 2-img scatter ok"
      if (all_ok and multi_ok) else f"FAIL (batch_eq={all_ok} multi_ok={multi_ok})")

# ---------- C: WEIGHTED grain mixing on real ArrayRecord (gs://) ----------
print("\n=== C: weighted grain mixing (controllable weights) ===")
BASE = "gs://uscentral2stuff/tmp/ar_arrecord_sample"
DSFILES = {
    "AGUVIS":   f"{BASE}/AGUVIS__b1024__s1088__part-aguvis-s0-00000-r00000.array_record",
    "Aria-UI":  f"{BASE}/Aria-UI__b1536__s1088__part-aria6-s0-00000-r00000.array_record",
    "ChartNet": f"{BASE}/FineVision__SynthChartNet__b1024__s1088__part-14-00000-r00000.array_record",
    "FormNet":  f"{BASE}/FineVision__SynthFormulaNet__b512__s1088__part-15-00000-r00000.array_record",
}
names = list(DSFILES)
sizes = {nm: len(gp.ArrayRecordDataSource([DSFILES[nm]])) for nm in names}
print("dataset sizes (rows):", sizes)
weights = [0.4, 0.3, 0.2, 0.1]   # arbitrary target mix, deliberately != size proportions
per_ds = []
for i, nm in enumerate(names):
    ds = gp.MapDataset.source(gp.ArrayRecordDataSource([DSFILES[nm]])).shuffle(seed=100 + i)
    ds = ds.map(lambda b, tag=nm: tag)   # tag instead of decoding (ratio test only)
    per_ds.append(ds)
mixed = gp.MapDataset.mix(per_ds, weights=weights)
N = 5000
from collections import Counter
draws = Counter(mixed[i] for i in range(N))
print("target weights:", dict(zip(names, weights)))
print("observed frac :", {nm: round(draws[nm] / N, 3) for nm in names})
size_frac = {nm: round(sizes[nm] / sum(sizes.values()), 3) for nm in names}
print("size-proportional would be:", size_frac)
ok_weight = all(abs(draws[names[i]] / N - weights[i]) < 0.03 for i in range(len(names)))
print("C RESULT:", "WEIGHTED MIX CONTROLLABLE (matches target weights, NOT sizes)" if ok_weight else "FAIL")
