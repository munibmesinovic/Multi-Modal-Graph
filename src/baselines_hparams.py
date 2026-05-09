BASELINE_HPARAMS = {
    "deepsurv": {
        "hidden_dims": [100, 100], "dropout": 0.4, "lr": 1e-4,
        "weight_decay": 1e-4, "epochs": 100, "patience": 15, "batch_size": 64,
    },
    "deephit": {
        "shared_dims": [100, 100], "cs_dims": [100], "dropout": 0.4,
        "lr": 1e-4, "weight_decay": 0.0, "alpha_rank": 0.1,
        "epochs": 100, "patience": 15, "batch_size": 64,
    },
    "dynamic_deephit": {
        "rnn_hidden": 100, "rnn_layers": 2, "cs_hidden": 100,
        "dropout": 0.4, "lr": 1e-4, "weight_decay": 1e-5,
        "alpha": 0.1, "gamma": 1.0, "epochs": 100,
        "patience": 15, "batch_size": 32,
    },
    "dysurv": {
        "encoded_features": 32, "lr": 1e-3, "weight_decay": 1e-4,
        "epochs": 100, "patience": 15, "batch_size": 64,
    },
}
