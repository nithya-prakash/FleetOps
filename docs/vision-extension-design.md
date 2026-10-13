# Optional vision extension (design only — **not implemented**)

**Decision**: not implemented in this round. The reasons are the dataset licence (below), the cost of a second model format and runtime that the
current evidence does not justify for the tabular model, and the fact that nothing in the repository could be validated end to end without
downloading a dataset we may not redistribute. This document records what would be built and what must be true first.

## Dataset licence check (2026-10-13)
The [MVTec AD page](https://www.mvtec.com/company/research/datasets/mvtec-ad) states the dataset is released under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 licence (CC BY-NC-SA 4.0): **non-commercial use only**, attribution required, derivatives under the same terms.
Mirrors repeat this ([Voxel51 dataset zoo](https://docs.voxel51.com/dataset_zoo/datasets_hf/mvtec_ad.html)); MVTec asks people unsure whether a use is
non-commercial to contact them. This is a reading of published terms, not legal advice.

Consequences for this project:
* The data must **never be committed or bundled** in the repository, images or CI artifacts. The extension would ship a download script that points to the
  official page and requires the user to accept the licence there; the files stay in a git-ignored directory.
* A portfolio demonstration is non-commercial, but any use of FleetOps with this data or models trained on it in a commercial setting would need MVTec's permission.
  Models trained on it (and their metrics) should be labelled with the licence in the registry metadata, and the ShareAlike clause treated as potentially applying to derived weights until clarified.
* Alternative for a commercially safe demo: synthetic defect images generated procedurally (we control the licence), at the price of weaker realism.

## What would be built
1. **Model**: an unsupervised anomaly detector trained on defect-free images only (e.g. a small convolutional autoencoder or a PatchCore-style memory bank on a frozen backbone), image-level score + threshold, optional anomaly map. Trained with the existing PyTorch training service (CPU).
2. **Model format**: an ONNX bundle (conv layers are not worth hand-writing in NumPy), signed/hashed exactly like the safetensors bundle, with the same `BundleMeta` fields plus input shape and preprocessing. This is the case in which `docs/edge-inference-benchmarks.md` says to revisit ONNX Runtime; it would add `onnxruntime` (~76 MB) to a separate `edge-vision` image target, leaving the tabular edge image unchanged.
3. **Registry/gate**: new evaluation suite (`image-benchmark-v1`) with image AUROC / F1 at threshold on a held-out set and the same fail-safe comparison rules as `sensor-benchmark-v1`; latency recorded per environment.
4. **Edge**: a `/predict-image` endpoint with strict size/format limits (decode bounded, no arbitrary file formats), same auth/rate limits; a camera feed adapter is out of scope (replay from a local folder).
5. **Drift**: embedding-statistics PSI on the backbone features instead of per-sensor PSI.
6. **Tests**: format load/verify, size limits, malformed image rejection, determinism of scores across the two architectures; benchmarks on the target CPU.

## Why it is not worth doing before the rest is finished
Image models are 1-2 orders of magnitude heavier than the current model: artifact size, download time, memory and CPU would stress exactly the paths
this round characterised (disk headroom, artifact download retry budget, single-core agent). The platform's limits for that case should be measured on
real target hardware first (`docs/edge-devices.md`), which has not been done.
