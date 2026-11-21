# Genetic Language Research — transfer kit

**Quick start:** `./setup.sh` sets up a local environment and verifies the stored results. See `ENVIRONMENT.md` for local, dev-container and cloud setup, and `STATUS.md` for what is verified.

Open `HANDOFF.md` for context and the scientific handoff. This folder is ready for a shared Git repository used by Claude Code and Codex.

**Reading and writing DNA:** `dnakit/` is the sequence layer — parse FASTA, GenBank and the UCI CSV, transcribe, translate, find reading frames, handle both strands, and write FASTA back out. `dnakit read|translate|orfs|revcomp|write|promoter`, or `import dnakit`. See `dnakit/README.md`; `make test` runs its 94 checks.

To start:
1. Extract ZIP to an empty folder and initialize a Git repo if needed.
2. Give Claude Code the prompt: `Read CLAUDE.md and HANDOFF.md. Reproduce experiment 002 first, then move to a valid public quantitative MPRA sequence-to-activity dataset. Use Codex for independent review. Return verifiable results, not a plan.`
3. If Claude and Codex run concurrently, use separate Git branches or worktrees.

Historical accuracy: experiment 001 reached 94.34% on a small E. coli dataset; experiment 002 did **not** show a benefit of masked pretraining. We do not yet have a verified quantitative human MPRA trained model.
