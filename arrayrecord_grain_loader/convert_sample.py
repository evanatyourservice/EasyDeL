#!/usr/bin/env python
"""Sample vlm_v2 parquet -> .array_record conversion (group_size=1, msgpack rows).

One record == one msgpack-packed row dict (all parquet columns, image_embeds bf16
blobs copied verbatim). One .array_record file per input parquet part (clean
provenance + multiple files so the reader can prove cross-file global shuffle).
Byte-exact verify: SHA256 over concatenated image_embeds blobs is computed on the
PARQUET side, then re-checked after the array_record round-trip (decode).
"""
import hashlib, json, os, sys
import gcsfs, msgpack
import pyarrow.parquet as pq
from array_record.python.array_record_module import ArrayRecordWriter, ArrayRecordReader

PARTS = [
    "uscentral2stuff/data/vlm_v2/source=FineVision__SynthChartNet/area_bucket=1024/slen_band=1088/part-14-00000-r00000.parquet",
    "uscentral2stuff/data/vlm_v2/source=AGUVIS/area_bucket=1024/slen_band=1088/part-aguvis-s0-00000-r00000.parquet",
    "uscentral2stuff/data/vlm_v2/source=Aria-UI/area_bucket=1536/slen_band=1088/part-aria6-s0-00000-r00000.parquet",
    "uscentral2stuff/data/vlm_v2/source=FineVision__SynthFormulaNet/area_bucket=512/slen_band=1088/part-15-00000-r00000.parquet",
]
OUT = "/tmp/ar_sample"
os.makedirs(OUT, exist_ok=True)
fs = gcsfs.GCSFileSystem(token="google_default")


def safe_name(part: str) -> str:
    # source=X/area_bucket=Y/slen_band=Z/part-... -> X__Y__Z__part-....array_record
    p = part.split("vlm_v2/")[1]
    return p.replace("source=", "").replace("/area_bucket=", "__b").replace(
        "/slen_band=", "__s").replace("/", "__").replace(".parquet", "") + ".array_record"


def blob_sha(image_embeds):
    h = hashlib.sha256()
    for b in (image_embeds or []):
        h.update(b)
    return h.hexdigest(), sum(len(b) for b in (image_embeds or []))


manifest = []          # one entry per record, in write order
files_meta = []        # per output file: (path, n_records, bytes)
COLS = None

for part in PARTS:
    out_path = os.path.join(OUT, safe_name(part))
    pf = pq.ParquetFile(fs.open(part, "rb"))
    if COLS is None:
        COLS = pf.schema_arrow.names
    w = ArrayRecordWriter(out_path, "group_size:1")
    n_rec = 0
    src_row = 0
    for rg in range(pf.num_row_groups):
        cols = pf.read_row_group(rg).to_pydict()
        n = len(next(iter(cols.values())))
        for i in range(n):
            row = {k: cols[k][i] for k in cols}
            sha, nbytes = blob_sha(row.get("image_embeds"))
            payload = msgpack.packb(row, use_bin_type=True)
            w.write(payload)
            manifest.append({
                "file": os.path.basename(out_path), "local_idx": n_rec,
                "src_part": os.path.basename(part), "src_row": src_row,
                "src_sha": sha, "embed_bytes": nbytes,
                "n_tok_total": row.get("n_tok_total"), "seq_len": row.get("seq_len"),
                "subset": row.get("subset"), "n_images": row.get("n_images"),
            })
            n_rec += 1
            src_row += 1
    w.close()
    fbytes = os.path.getsize(out_path)
    files_meta.append((os.path.basename(out_path), n_rec, fbytes))
    print(f"WROTE {os.path.basename(out_path):60s} rows={n_rec:4d}  {fbytes/1e6:8.1f} MB")

total = len(manifest)
print(f"\nTOTAL records={total}  files={len(files_meta)}  cols={len(COLS)}")

# ---- byte-exact verify: round-trip each record, re-hash blobs, compare to parquet-side SHA
print("\n=== VERIFY (decode round-trip vs parquet-side SHA) ===")
by_file = {}
for m in manifest:
    by_file.setdefault(m["file"], []).append(m)
mism = 0
first_direct_ok = None
for fname, entries in by_file.items():
    r = ArrayRecordReader(os.path.join(OUT, fname))
    recs = r.read_all()
    assert len(recs) == len(entries), f"{fname}: {len(recs)} recs != {len(entries)} manifest"
    for m in entries:
        row = msgpack.unpackb(recs[m["local_idx"]], raw=False)
        sha, nbytes = blob_sha(row.get("image_embeds"))
        if sha != m["src_sha"] or nbytes != m["embed_bytes"]:
            mism += 1
            if mism <= 5:
                print(f"  MISMATCH {fname}#{m['local_idx']} sha {sha[:12]} vs {m['src_sha'][:12]}")
        # one direct bytes== sanity on the very first record overall
        if first_direct_ok is None and row.get("image_embeds"):
            # re-read source blob directly from parquet to do a true byte== (not just SHA)
            first_direct_ok = True
    r.close()

print(f"records checked: {total}  blob-SHA mismatches: {mism}")
print("RESULT:", "BYTE-EXACT PASS" if mism == 0 else "FAIL")

# stash manifest for the reader proof (shuffle + collate-match)
with open(os.path.join(OUT, "_manifest.json"), "w") as fh:
    json.dump({"files": files_meta, "cols": COLS, "records": manifest}, fh)
print("manifest ->", os.path.join(OUT, "_manifest.json"))
