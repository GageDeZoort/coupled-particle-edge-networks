# MNISTSuperpixels readout study

Graph classification (digit 0–9). Twin of `scans/pascal/` except the question is
**how edges enter the graph readout**, not node vs node+boundary co-training.

Pipeline matches `~/hp-transfer-gts` (`MNISTSuperpixels`, official train split
→ 90/10 train/val with `split_seed=42`, official test, metric = accuracy).
No LapPE in this first CPEN pass (hp-transfer `encoder=linear`).

## Readout arms

| `READOUT`     | Graph logit                         | Status        |
|---------------|-------------------------------------|---------------|
| `node`        | pooled \(z_X\) only                 | wired         |
| `node+edge`   | \((z_X + z_E)/\sqrt{2}\) after pool | wired         |
| `attn`        | class-token MHSA over nodes+edges   | wired (`--readout-mode attn`) |

\(z_X = X^{(L)} W_X / \sqrt{D}\), \(z_E = E^{(L)} W_E / \sqrt{D}\), then a
mask-aware pool (same energy / incidence pool as stock graph mode). There is
**no** per-edge CE: MNIST has only a graph label.

`--dataset mnist-sp` is wired in `scans/common/test_cpen.py`. Download/build
the incidence cache first, then submit. Checkpoint monitor is `val_acc`.

## Commands

```bash
# raw PyG MNISTSuperpixels (login node if compute has no outbound net)
python scans/mnist/download_mnist_graphs.py \
  --data-root /scratch/gpfs/BHANIN/jdezoort/datasets
# or: sbatch scans/mnist/download_mnist_graphs.slurm

sbatch scans/mnist/build_mnist_graphs.slurm

DRY_RUN=1 bash scans/mnist/submit_readout.sh
READOUTS=node,node+edge bash scans/mnist/submit_readout.sh
READOUTS=attn WIDTHS=512 ETAS=0.25 bash scans/mnist/submit_readout.sh
```
