# Environments

The same pinned `requirements.txt` drives all three environments, so results
match across them. Experiment 001 reproduces exactly; `scripts/verify.py`
checks this on any machine.

## Local (macOS or Linux)

```bash
git clone https://github.com/yegerman/genome_codex.git
cd genome_codex
./setup.sh
```

`setup.sh` finds Python 3.11 or newer, creates `.venv`, installs the pinned
packages, prints which accelerator torch can use, and re-runs experiments 001
and 002 in a temporary copy to confirm the stored results. Stored results are
never modified. Measured in the cloud container: about 2 minutes from empty.

On macOS without Python 3.11+: `brew install python@3.13`.
On an Apple Silicon Mac, torch reports `mps`; on a machine with an NVIDIA GPU
it reports `cuda`. Either is useful for the pretrained-model experiment.

Day-to-day commands:

| command | what it does |
|---|---|
| `make verify` | re-run 001 and 002 and compare with stored results |
| `make audit` | leakage and fold-stability audit, about 1 minute |
| `make verify-dataset` | download the official UCI file and compare |
| `make clean` | delete `.venv` |

Windows: use WSL, or open the repo in VS Code with the Dev Containers
extension, which runs `setup.sh` automatically from `.devcontainer/`.

## Cloud (Claude Code on the web)

Environment name: `genome_code` (personal).

- Allowed hosts: `archive.ics.uci.edu`, `huggingface.co`,
  `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`,
  `www.ncbi.nlm.nih.gov`, `ftp.ncbi.nlm.nih.gov`, plus package registries.
- Setup script installs `genome_codex/requirements.txt` at session start.
- Limits: 4 CPUs, no GPU. Fine for baselines and audits, not for running a
  large pretrained genomic model.

## Not yet covered

- Hosts for MPRA data outside NCBI, and `download.pytorch.org`, are not on
  the cloud allowlist. Add them when an experiment needs them.
- The pretrained-model experiment in `experiments/004` has run locally only
  (Apple M5 Pro, about 1.5 minutes on CPU). Its first run downloads about
  224 MB of model weights from `huggingface.co` into `~/.cache/huggingface`.
  Its model code needs transformers 4.x, which is why `requirements.txt` pins
  4.57.6.
