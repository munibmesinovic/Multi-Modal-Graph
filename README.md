# MM-GraphSurv: Multi-Modal Graph Neural Network for Competing Risks Prediction

Official implementation of MM-GraphSurv, a spatio-temporal graph neural network for multi-modal survival analysis with competing risks.

## Repository Structure
```
├── 00_setup.py                 # Environment setup and imports
├── 01_visits.py                # Patient visit preprocessing
├── 02_radiology.py             # Radiographic report embeddings (Clinical-Longformer)
├── 03_icd.py                   # ICD code embeddings (Word2Vec + cosine similarity)
├── 04_labs.py                  # Laboratory time-series processing
├── 05_vitals.py                # Vital signs time-series processing
├── 06_transform_split.py       # Data transformation and train/val/test split
├── 07_save_final.py            # Save processed data
├── train_graphsurv_mcmed.py    # Model training script
└── requirements.txt            # Dependencies
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing (run sequentially)
```bash
python 00_setup.py
python 01_visits.py
python 02_radiology.py
python 03_icd.py
python 04_labs.py
python 05_vitals.py
python 06_transform_split.py
python 07_save_final.py
```

### 2. Model Training
```bash
python train_graphsurv_mcmed.py
```

## Key Components

- **EMA-smoothed dynamic adjacency**: Temporal stability via exponential moving average (α=0.7)
- **ICD embeddings**: Word2Vec (Skip-gram, d=128) with cosine similarity graph
- **Radiographic embeddings**: Clinical-Longformer with Gaussian similarity kernel (τ=2.0)
- **Hierarchical attention**: Learnable cross-modal attention matrices

## Datasets

Evaluated on MIMIC-IV, eICU, MC-MED, PBC2, and SUPPORT. See paper for preprocessing details.

## Citation
```

```
```

---
