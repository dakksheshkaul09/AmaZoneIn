# Business Entity Resolution - Execution Guide

## Hardware & Environment Constraints
This pipeline is explicitly engineered to run on consumer hardware (16 GB RAM) by decoupling candidate generation from feature extraction and utilizing lockstep chunking.

## Setup
1. Create a Python 3.9+ virtual environment.
2. Run `pip install -r requirements.txt`.
3. Place the test datasets in `dataset/test/` (i.e., `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`).

## End-to-End Execution
To regenerate the output files (`candidate_pairs.tsv` and `matching_results.tsv`), run the main submission pipeline:

```bash
python src/v21_lgb_submission_runner.py