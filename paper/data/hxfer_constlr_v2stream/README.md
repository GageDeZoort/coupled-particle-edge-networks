# hxfer const-LR v2stream — Lambda → Della organization

Adam const-LR η₀×D @ 10k steps, single-pass n_train=5.12M, k★=6, R★=0.1,
B_eff=512, L=4, d_h=32. Host: `ubuntu@193.122.155.21`.

## Layout

| Path | Contents |
|---|---|
| `paper/data/hxfer_constlr_v2stream/{upool,apool}/logs/` | stdout + master.log |
| `paper/data/hxfer_constlr_v2stream/{upool,apool}/summaries/` | manifests + metrics CSV |
| `/scratch/.../hxfer_constlr_v2stream/{upool,apool}/runs/` | symlinks → SWEEP run dirs |
| `/scratch/.../hxfer_constlr_v2stream/{upool,apool}/parquets/` | symlinks → metrics parquets |
| `/scratch/.../hxfer_constlr_v2stream/{upool,apool}/checkpoints/` | symlinks → best/last.ckpt |
| `/scratch/.../jetclass_output/adam/sweep_lr/*hxfer-*-v2stream*` | canonical pulled artifacts |

## Pull scripts

- `scans/jets/pull_hxfer_upool_eta_constlr_lambda.sh`
- `scans/jets/pull_hxfer_apool_eta_constlr_lambda.sh`
- `scans/jets/pull_hxfer_logs_lambda.sh`

## Paper figures

- Uniform (Nature): `paper/figures/fig_hparam_transfer_proof_upool10k.{png,pdf,eps}`
- Attn (prelim): `paper/figures/fig_hparam_transfer_proof_apool10k.{png,pdf,eps}`
- Metrics: `paper/data/hxfer_{upool,apool}_eta_constlr.csv`
