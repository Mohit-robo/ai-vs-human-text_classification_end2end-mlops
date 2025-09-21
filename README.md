# ai-vs-human-text_classification_end2end-mlops

    mlops-svm-text-classifier/
    │
    ├── data/                  # datasets (raw & processed)
    ├── notebooks/             # Jupyter notebooks (EDA etc.)
    │   └── 01_eda.ipynb
    │
    ├── src/                   # source code
    │   ├── __init__.py
    │   ├── data.py            # data loading & preprocessing
    │   ├── train.py           # training + RayTune + MLflow logging
    │   ├── evaluate.py        # evaluation utilities
    │   └── serve.py           # FastAPI inference service
    │
    ├── configs/               # config files
    │   └── config.yaml
    │
    ├── artifacts/             # saved models, plots, confusion matrices
    │
    ├── requirements.txt
    ├── README.md
    └── run.sh                 # shell script for quick run
