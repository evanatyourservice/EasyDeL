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

## Data-wait benchmark (the acceptance bar: loader >= step rate, no starvation)

`bench_throughput.py` — full production op chain (msgpack-decode → per-batch pack
collate) reading `.array_record` straight from `gs://` under TRUE global shuffle
(worst-case random access), grain parallel prefetch, single host, worker_count=0.
gs:// reads work NATIVELY in array_record (no gcsfuse): source open 0.74s, cold random
1.8MB record read 0.29s single-thread.

Sustained ~35–37 rows/s, ~195 MB/s (read/decode-bound; flat across num_threads 4–32):

| batch rows | ms/batch | data-wait/step @22s | headroom |
|---|---|---|---|
| 8  | 215 | 0.00s | 102× |
| 16 | 431 | 0.00s | 51× |
| 32 | 912 | 0.00s | 24× |

Conclusion: data-wait ≈ 0 with 24–102× headroom over the ~22s step even single-host /
conservative. Starvation would need ~770 rows/step on ONE host; production = 16 hosts,
each loader feeds only its own IndexSampler shard, so per-host headroom holds (aggregate
read BW scales ~16×). worker_count>0 (multiprocess) is available for more if ever needed.
Caveat: 195 MB/s is the head CPU VM's plateau; TPU pod hosts likely read faster. MFU
(packing density) is secondary per Evan and not optimized here.

## Production wiring recipe (Option A, proven by the benchmark)

```python
src  = grain.python.ArrayRecordDataSource(gs_files)          # native gs://, random access
samp = grain.python.IndexSampler(num_records=len(src),
          shard_options=grain.python.ShardOptions(process_index, process_count, drop_remainder=True),
          shuffle=True, seed=..., num_epochs=...)             # TRUE global shuffle + per-host shard
dl   = grain.python.DataLoader(data_source=src, sampler=samp,
          operations=[MsgpackDecode(),                        # bytes -> row dict (new MapTransform)
                      grain.python.Batch(B, drop_remainder=True, batch_fn=collate)],  # Batch-then-collate
          worker_count=W, worker_buffer_size=2,
          read_options=grain.python.ReadOptions(num_threads=16, prefetch_buffer_size=64))
```
`grain.python.Batch` accepts `batch_fn`, so VL packing (many-rows→one-batch) fits without
a bespoke stage — unlike the existing per-row grain path (`base_trainer.py:5034-5044`).

## Production wiring (committed: bbe0064b)

Wired into the trainer so the VL pack can run through grain ArrayRecord by config alone:

- `training_configurations.py`: new fields — `arrayrecord_train_files` / `arrayrecord_eval_files`
  (path / glob / list), `arrayrecord_num_threads` (16), `arrayrecord_prefetch_buffer` (64),
  `arrayrecord_worker_count` (0). Shuffle/seed/epochs/sharding reuse existing fields
  (`shuffle_train_dataset`, `shuffle_seed_train`, `num_train_epochs`, `grain_shard_index/count`).
- `trainers/utils.py`: `MsgpackDecode(pygrain.MapTransform)` — record bytes → row dict (lazy msgpack).
- `base_trainer.py`: `configure_dataloaders()` dispatches to new `_configure_arrayrecord_dataloader()`
  when `arrayrecord_train_files` is set (checked before use_grain/tfds). It builds
  `ArrayRecordDataSource(expand(files))` → `IndexSampler(shuffle, ShardOptions(proc_idx, proc_count))`
  → `DataLoader(ops=[MsgpackDecode(), Batch(B, drop_remainder=True, batch_fn=self.data_collator)],
  ReadOptions(num_threads, prefetch_buffer_size))`. `data_collator` (the VL packed-embeds collator,
  e.g. EmbedsWindowPacker→collate_packed_embeds) is the per-batch `batch_fn`, so the loader yields
  collated dicts; the loop's `_apply_user_data_collator` no-ops on a dict (no double collation),
  then `_purify_batch` strips non-array fields. Per-host step count = `(len//shard_count//B)*epochs`.
- `pyproject.toml`: `array_record; platform_system=='Linux'` + `msgpack>=1.0.0`.

Usage: set `data_collator=<VL collator>` + `arrayrecord_train_files=[gs://…*.array_record]`; no other
change. Validation: syntax (py_compile) + glue verified vs real signatures + the core pipeline is the
benchmark above. Remaining: full distillation trainer smoke on v4-128 to read the trainer's own
data_collection_time (confirmatory; the benchmark already establishes data-wait≈0). Resume/seek
(grain checkpointing) and full-vlm_v2 conversion are the next iteration.

## Completeness pass (per Evan/orch) — checklist

| # | Dimension | Status | Evidence / note |
|---|-----------|--------|-----------------|
| 1 | **Weighted mixture** | ✅ COVERED | Re-architected to grain best practice: each dataset = its OWN ArrayRecord set → `MapDataset.source().shuffle()` → `MapDataset.mix(weights)`. Proven controllable: target 0.4/0.3/0.2/0.1 → observed **exactly** 0.4/0.3/0.2/0.1 (size-proportional would be 0.18/0.18/0.16/0.49). Weights plumbed via `arrayrecord_train_datasets={name:files}` + `arrayrecord_mixture_weights={name:w}` (mirrors the existing `mixture_weights`). NOT a single concatenated uniform shuffle. |
| 2 | **Multi-image** | ✅ COVERED | `FineVision__mimic_cgd` (2 img/row): 24 rows, **0** per-image-SHA mismatches; collate scatters **multiple spans/sample** — 2-img row = 98 placeholders = 98 embed tok, batch (392,5120) byte-exact vs parquet. n_images up to 3 present (AgentNet, GUI-Net) — same per-image list mechanism. |
| 3 | **Packed sequences** | ✅ (per-batch) | `data_collator` = EmbedsWindowPacker→collate_packed_embeds (per-batch packing) runs as the grain `batch_fn`. MFU/density reported for visibility only; not over-optimized (data-faster-than-training is the bar). Knobs if ever needed: larger packing unit / soft length-grouping per bucket. |
| 4 | **group_size vs throughput** | ✅ group_size=1 holds | group_size=1 (max random access / global shuffle). Sustains ~30–37 rows/s, ~195 MB/s from gs:// → 24–102× over the ~22s step, data-wait≈0. No need for grouping; if reads ever lag at full fan-out, group_size 8–16 is the fallback (trades shuffle granularity for fewer seeks). |
| 5a | Per-host sharding | ✅ COVERED | `mixed.slice(slice(process_index, None, process_count))` — deterministic, disjoint strides verified. |
| 5b | Index resume | ✅ native | grain `IterDataset` is index/checkpointable (grain.checkpoint) — seek by index, no row materialization. (Wiring exposes it; resume hook is a follow-up knob.) |
| 5c | Determinism / seed | ✅ COVERED | per-dataset `.shuffle(seed=shuffle_seed_train + i)`; same seed → same global order. |
| 5d | Prefetch / threads | ✅ COVERED | `to_iter_dataset(ReadOptions(num_threads, prefetch_buffer_size))` (config: `arrayrecord_num_threads`/`arrayrecord_prefetch_buffer`). |

### Weighted-mix architecture (final)
```
per dataset:  MapDataset.source(ArrayRecordDataSource(files)).shuffle(seed+i)
mixture:      MapDataset.mix(per_ds, weights=[w_i])          # controllable weights
per host:     .slice(slice(process_index, None, process_count))
epochs:       .repeat(num_train_epochs)                       # mix len = min_i(size_i / w_i)
decode+batch: .map(MsgpackDecode()).batch(B, drop_remainder=True, batch_fn=data_collator)
drive:        .to_iter_dataset(ReadOptions(num_threads, prefetch_buffer_size))
```
**Conversion layout consequence:** the full vlm_v2 conversion writes **one ArrayRecord set per dataset** (per `source=`), NOT one concatenated set — so weights stay controllable. `mix` length is bounded by the most-constrained `size_i/w_i`; `repeat` cycles. group_size=1, msgpack rows, bf16 blobs verbatim (as in the sample convert).

## Corners cut / proven-vs-assumed (honest accounting)

PROVEN (empirical):
- Byte-exact convert/round-trip (incl multi-image), weighted-mix controllability, multi-span
  scatter, the full convert→read→collate pipeline, the uncollated-list collation contract —
  all as REPO TESTS (see below) + on real gs:// sample data.
- gs:// native ArrayRecord reads; data-wait≈0 in the real trainer (data_collection_time 0.608s
  vs 607s step); a real KD step with sane loss through the wired path.

ASSUMED / DEFERRED / SIMPLIFIED:
- SCALE: all proofs + smokes use a 4-file SAMPLE (≈410 rows; one parquet part per source) +
  one multi-image part. The full ~1.4M-row vlm_v2 conversion is NOT done (Evan-gated next).
- THROUGHPUT is single-host: ~195 MB/s / 24–102× headroom measured on the head CPU VM only.
  No production multi-host (16-host) aggregate-throughput test, and not on a TPU-pod host
  (likely faster net). worker_count=0 (in-process threads); multiprocess prefetch not measured.
  Expectation: per-host headroom holds (each host reads only its shard); not empirically at scale.
- MFU/packing density: per-batch EmbedsWindowPacker, NOT tuned or measured (Evan: secondary).
- RESUME/SEEK: grain IterDataset is checkpointable, but no explicit checkpoint/resume hook is
  wired — `_ReiterableDataLoader` restarts fresh per iter (no mid-epoch seek). Needed for the
  full run's resume-from-step; deferred.
- EVAL path (`arrayrecord_eval_datasets`) is wired but only the TRAIN path was smoke-tested.
- MIX EPOCH LENGTH = min_i(size_i / weight_i): a dataset whose weight is large relative to its
  size becomes the binding constraint (epoch ends when it would exhaust). Full-run weights must
  account for this (or rely on `.repeat`). Not validated at full scale.
- NaN-grad after step 1 on the bare seq2048/lr1e-4 config: shown to be path-independent
  (config/numerics, not the loader) via the control test; the NaN itself is NOT root-caused
  (training-stability, Evan's domain — production uses the klfix2-stabilized image/config).

## Repo test coverage (libs/easydel/tests/data/test_arrayrecord_grain.py — 5 passing)

| Test | Covers |
|------|--------|
| test_batch_byte_exact_arrayrecord_vs_parquet | (i) collated batch identical arrayrecord vs parquet |
| test_weighted_mix_proportions_controllable   | (ii) grain mix samples by weights, not sizes |
| test_multi_image_scatter                      | (iii) n_images≥2 round-trip + multi-span scatter |
| test_convert_read_collate_pipeline            | (iv) parquet→.array_record→source→decode→collate |
| test_loader_yields_uncollated_lists           | (v) batch_fn=list contract (double-collation regression) |

Self-contained: synthetic fixtures (tiny dims), `importorskip` for array_record/grain/msgpack/
ml_dtypes — runs in CI with no GCS/TPU/staged data. NOT unit-tested (smoke/ad-hoc only): the
in-trainer 27B KD step (smoke), gs:// reads (needs creds), multi-host shard disjointness
(ad-hoc in mapdataset_pipeline.py).
