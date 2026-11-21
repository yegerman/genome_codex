.PHONY: setup test verify audit verify-dataset clean

setup:
	./setup.sh --no-verify

# Unit tests for dnakit, the DNA reading/writing library. Seconds, no network.
test:
	.venv/bin/python -I tests/test_dnakit.py

verify:
	.venv/bin/python -I scripts/verify.py

# Leakage and fold-stability audit (about 1 minute on a laptop CPU).
audit:
	cd experiments/003/leakage_audit && ../../../.venv/bin/python -I audit_grouping.py && ../../../.venv/bin/python -I audit_fold_stability.py

# Re-download the official UCI file and compare it to the local CSV.
verify-dataset:
	cd experiments/003/dataset_verification && ../../../.venv/bin/python -I fetch_uci_promoters.py

clean:
	rm -rf .venv
