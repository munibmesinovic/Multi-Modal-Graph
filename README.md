# GraphSurv: Multi-Modal Graph Neural Network for Competing Risks Prediction

This repository contains the official implementation of **GraphSurv**, a spatio-temporal graph neural network designed for multi-modal survival analysis with competing risks. The model is evaluated on MC-MED, MIMIC-IV, eICU, PBC2, and SUPPORT datasets.

## Repository Structure

- `data_preprocessing.py`: Processes the MC-MED dataset into time-series, static, ICD code, and radiographic embeddings.
- `train_graphsurv_mcmed.py`: Trains the GraphSurv model on the processed MC-MED dataset.

## Setup

To set up the environment, run:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing

```bash
python data_preprocessing.py
```

### 2. Model Training

```bash
python train_graphsurv_mcmed.py
```

## Citation

If you use this code, please cite our paper (link to be updated):

```
@article{your2024graphsurv,
  title={GraphSurv: Multi-Modal Graph Neural Network for Competing Risks Survival Prediction},
  author={Author et al.},
  journal={},
  year={2024}
}
```

## License

This project is licensed under the MIT License.
