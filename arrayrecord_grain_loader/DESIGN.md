# ArrayRecord + grain dataloader for the VL pack — design note

Branch: `arrayrecord-grain-loader` off `upstream/vnext` (EasyDeL fork). Task #1761.

## Why (the problem this solves)

The repacked VL pack (`vlm_v2`) is consumed today through the **ShardedDataSource**
path, not grain:

- `ParquetShardedSource` (one parquet file = one shard) → optional transforms →
  `_train_source` → `BaseTrainer._create_dataloader_from_source`
  (`base_trainer.py:4935`). That helper iterates shards **in file order** and slices
  fixed-size batches. Its `shuffle` arg is a no-op ("currently not implemented for
  sources", docstring line 4955).
- The only shuffle available on this path is `ShuffledShardedSource`
  (`data/transforms/source.py:167`): it collapses the source into one synthetic shard
  and draws rows through a **fixed-size streaming reservoir**. That is an *approximate*
  shuffle — decorrelation is bounded by `buffer_size`, rows never travel further than
  one reservoir window from their on-disk position, and a global permutation is
  impossible without buffering the whole dataset in RAM.

grain already has a *true* random-access path in the trainer
(`_create_grain_dataloader`, `base_trainer.py:4993`): `grain.IndexSampler(shuffle=True)`
permutes the full `[0, N)` index space and `grain.ShardOptions` partitions it per host.
But it only fires when `_train_source is None` **and** the dataset is a HF
`Dataset`/`IterableDataset` — the VL pack never reaches it. And the existing
`HFDataSource` bridge (`trainers/utils.py:2144`) is a *fake* random-access source
(`__len__` returns 1e10, `__getitem__` ignores the index and just calls `next()`), so
even that path can't do a real global shuffle for streamed data.

**ArrayRecord fixes this at the storage layer.** Each row becomes an independently
addressable record, so the dataset is genuinely random-access: `len = total rows`,
`__getitem__(i) = row i` in O(1). That is exactly the substrate
`grain.IndexSampler(shuffle=True)` needs to deliver a *full-dataset* shuffle with
constant memory, plus clean per-host sharding and index-based (materialization-free)
checkpoint resume.

## What ArrayRecord is

Record-addressable container (Riegeli-based). A file holds N records; the reader builds
an index so any record is fetchable by global offset. `group_size=1` → one row per
record (max addressability; required for row-level global shuffle).

- `grain.sources.ArrayRecordDataSource(paths)` wraps a list of `.array_record` files
  into one `RandomAccessDataSource` (`len` = sum of records, `__getitem__(i)` = raw
  record `bytes`). This is the real thing the fake `HFDataSource` only imitates.
- Dep status: `grain~=0.2.11` is already a **core** dep (`libs/easydel/pyproject.toml:46`).
  The `array_record` package (writer + the C++ reader grain's source binds to) is **not**
  declared and is **Linux-only** (no macOS wheel) — must be added as a dep, and all
  conversion/read work runs on Linux (cluster/pod), not the laptop.

## Record format (byte-exact with the parquet path)

One record = `msgpack` of the row dict, keys mirroring exactly what
`ParquetShardedSource` yields (`_table_to_rows`, `sources/base.py:401` → `{col: v[i]}`):

| key | type | note |
|---|---|---|
| `input_ids` | list[int] | token ids |
| `labels` | list[int] | loss targets |
| `image_embeds` | list[bytes] | **bf16 blobs, copied verbatim** — no re-encode |
| `embed_n_tok` | list[int] | per-image token counts |
| `embed_dim` | int | embed width (also inferrable from blob bytes) |
| (+ any text-row fields present in the pack schema) | | |

`msgpack.packb(row, use_bin_type=True)` / `unpackb(..., raw=False)` round-trips
`bytes` losslessly. **The bf16 image-embed blobs are passed through untouched**, so the
decoded array is bit-identical to the parquet path — verified by SHA over the
concatenated blobs (conversion deliverable #2).

## Pipeline / where it plugs in

```
.array_record files
  → grain.sources.ArrayRecordDataSource           (RandomAccessDataSource: len, __getitem__→bytes)
  → grain.IndexSampler(shuffle=True, seed, shard_options)   (TRUE global permutation + per-host shard)
  → grain.DataLoader(operations=[
        MsgpackDecode(),         # bytes → row dict   (new, stateless MapTransform)
        ToNumpy(),               # reuse trainers/utils.py:2276
        <collate>,               # see "collator bridge"
        grain.Batch(batch_size, drop_remainder=True),
    ])
```

This mirrors the existing `_create_grain_dataloader` almost exactly — the only new
pieces are the real ArrayRecord source and the msgpack-decode op.

**Integration seam (two options):**

- **Option A — grain-native (recommended).** Detect `*.array_record` `data_files` and
  build the grain `DataLoader` above directly, bypassing `_train_source`. Keeps the true
  IndexSampler global shuffle — the whole point. New code: an `ArrayRecordDataSource`
  thin wrapper (or use grain's directly) + `MsgpackDecode` op + a small branch in the
  dataloader-selection logic.
- **Option B — ShardedDataSource wrapper.** Wrap ArrayRecord as a `ShardedDataSource`
  (one shard per file; `open_shard_at_row` seeks by record index) so it flows through the
  existing `_train_source` path, transforms, and the `BatchHomogeneousMixedShardedSource`
  I added for the two-compile work. **But** that path's only shuffle is the reservoir —
  it throws away ArrayRecord's main advantage. Use only if reusing the transform stack
  matters more than shuffle quality.

Recommend **A** for the shuffle win; B remains available if we want ArrayRecord under the
existing mix/transform machinery.

## Collator bridge (the one real fork in the road)

The VL collator is `collate_packed_embeds` (`data/transforms/collators.py`), driven by a
**stateful** `EmbedsWindowPacker` that carries leftover rows *across* batch boundaries to
densify packed windows. grain's model is stateless per-element `MapTransform` + `Batch`,
which does not host a cross-batch packer cleanly. Options:

1. **Per-batch packing (simple, recommended first).** `grain.Batch(B)` groups B rows → a
   batch-level transform runs an `EmbedsWindowPacker` *within* that batch and calls
   `collate_packed_embeds`. Loses cross-batch carryover (a few partial windows per batch)
   but keeps within-batch packing and is fully grain-native. This is what the proof
   (deliverable #3) demonstrates.
2. **Full-fidelity packing.** Keep `EmbedsWindowPacker` stateful in an outer loop and let
   grain own only shuffled row *delivery*. Higher MFU (denser windows) but reintroduces a
   stateful stage outside grain and complicates resume.

**This is the fork worth your/Evan's input:** is the cross-batch packing density (option 2)
worth keeping a stateful packer outside grain, or is per-batch packing (option 1) good
enough? It's an MFU-vs-simplicity call; the data correctness is identical either way. I'll
build #1 for the proof and we decide before production wiring.

(Non-VL/text rows already work as plain row dicts; the all-language drop path lands via the
`drop_empty_embeds` collator branch from the two-compile work.)

## Distributed batching & resume

- **Sharding:** `grain.ShardOptions(shard_index=process_index, shard_count=process_count,
  drop_remainder=True)` partitions the global index per host; same `seed` → consistent
  global permutation, each host draws its disjoint slice. Identical wiring to the existing
  grain path.
- **Resume:** IndexSampler is index-based, so checkpoint resume seeks by global index with
  **no row materialization** — strictly better than the reservoir's replay-by-fast-forward.

## Deliverables (this branch)

1. This design note.
2. Sample conversion: a handful of `vlm_v2` parquet parts (mixed buckets) → `.array_record`
   (`group_size=1`, one msgpack row/record, ~hundreds-MB shards), reusing the repack
   streaming-read harness; **byte-exact** embed verify (SHA over blobs, parquet vs decoded).
3. Working grain reader on the sample: `ArrayRecordDataSource` → `IndexSampler(shuffle=True)`
   → msgpack-decode → collate → correct batch; proves **global, cross-file/cross-dataset
   sample-level shuffle** and a batch that matches the parquet path.

Then iterate to full `vlm_v2` conversion + production wiring (Option A + collator choice).

## Results (sample run, cluster head ray_docker, :vnext image)

Tooling already in the image: `grain` 0.2.17, `array_record`, `msgpack`, `pyarrow`,
`ml_dtypes`. Sample = 4 parquet parts spanning 4 datasets × 3 area_buckets:
SynthChartNet(b1024), AGUVIS(b1024), Aria-UI(b1536), SynthFormulaNet(b512).

**convert_sample.py** → 4 `.array_record` files, 410 records, ~340–524 MB each
(`group_size:1`, one msgpack row/record, all 18 columns, image_embeds bf16 verbatim).
Byte-exact verify: **0/410** blob-SHA mismatches across parquet→msgpack→array_record→decode.

**read_proof.py**
- Proof 1 (global shuffle): `IndexSampler(shuffle=True, seed=42)` over
  `ArrayRecordDataSource([4 files])` yields a **true permutation of [0,410)**, interleaving
  all 4 datasets (verified `sorted(keys)==range(N)`; all sources appear in the first draws).
- Proof 2 (batch == parquet path): a 6-row shuffled batch decoded from ArrayRecord and the
  *same* rows read straight from parquet, both run through `collate_embeds_pack`, produce
  **byte-exact-identical** tensors — `input_ids/attention_mask/labels (6,1069)`,
  `image_embeds (2363,5120) bf16`, `image_embed_positions/mask`, `image_grid_thw`,
  `n_real_embeds`. The bf16 `image_embeds` equality confirms the blobs round-trip and
  scatter identically.

Note: `image_token_id=248056` is auto-detected from the data. `collate_embeds_pack`'s
`max_total` is the **embed-row capacity** (Σ `embed_n_tok` over the batch), not the seq len.
