"""Run the ZU-PWD pipeline, then score every converted dataset: python src/run_all.py

The external datasets must be converted first with src/datasets/<name>.py (see README).
"""
import clean
import coupling_test
import explore
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
