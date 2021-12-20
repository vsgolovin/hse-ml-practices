from pathlib import Path

RAW_PATH = "data/raw"              # raw datasets
INTERIM_PATH = "data/interim"      # cleaned-up raw data
PROCESSED_PATH = "data/processed"  # datasets for training / testing model
PLOT_PATH = "reports/figures"
SUBMISSION_PATH = "reports/data"   # submission.csv for Kaggle
SEED = 42                          # random seed


rule all:
    input:
        Path(PLOT_PATH, "feature_importance.png"),
        Path(PLOT_PATH, "roc.png"),
        Path(SUBMISSION_PATH, "submission.csv")

rule eda:
    input:
        Path(RAW_PATH, "train.csv"),
        Path(RAW_PATH, "test.csv")
    output:
        Path(INTERIM_PATH, "train.csv"),
        Path(INTERIM_PATH, "test.csv")
    shell:
        "python src/eda.py --input_dir={RAW_PATH} --output_dir={INTERIM_PATH}"

rule feature_engineering:
    input:
        Path(INTERIM_PATH, "train.csv"),
        Path(INTERIM_PATH, "test.csv")
    output:
        Path(PROCESSED_PATH, "train.csv"),
        Path(PROCESSED_PATH, "test.csv")
    shell:
        "python src/feature_engineering.py"
        " --input_dir={INTERIM_PATH}"
        " --output_dir={PROCESSED_PATH}"

rule run_classifier:
    input:
        Path(PROCESSED_PATH, "train.csv"),
        Path(PROCESSED_PATH, "test.csv")
    output:
        Path(PLOT_PATH, "feature_importance.png"),
        Path(PLOT_PATH, "roc.png"),
        Path(SUBMISSION_PATH, "submission.csv")
    shell:
        "python src/model.py"
        " --input_dir={PROCESSED_PATH}"
        " --output_dir={SUBMISSION_PATH}"
        " --plot_dir={PLOT_PATH}"
        " --seed={SEED}"
