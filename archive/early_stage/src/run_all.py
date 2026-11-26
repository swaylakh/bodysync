"""Run the ZU-PWD pipeline, then score every converted dataset: python src/run_all.py

The external datasets must be converted first with src/datasets/<name>.py (see README).
The final push and the later checks have their own commands, because the locked-test runs
are meant to be run once (each writes its own results file):
  python src/final_eval.py --round N | --final | --final-v2
  python src/new_datasets.py
  python src/audit.py
"""
import clean
import coupling_test
import explore
import feature_test
import features
import validate_datasets
import validate_zupwd

if __name__ == "__main__":
    clean.clean()
    features.main()
    explore.main()
    coupling_test.main()
    validate_zupwd.main()
    validate_datasets.main()
    feature_test.main()
