search_space = [
    {"name": "learning_rate", "type": "continuous", "domain": (0.001, 0.1)},
    {"name": "layer_1", "type": "discrete", "domain": [64, 128, 256, 512]},
    {"name": "layer_2", "type": "discrete", "domain": [64, 128, 256, 512]},
    {"name": "layer_3", "type": "discrete", "domain": [64, 128, 256, 512]},
    {"name": "keeps_prob", "type": "discrete", "domain": [0.1, 0.2, 0.3, 0.4, 0.5]},
    {"name": "beta_1", "type": "continuous", "domain": (0.9, 0.9999)},
    {"name": "beta_2", "type": "continuous", "domain": (0.9, 0.9999)},
    {
        "name": "batch_size",
        "type": "discrete",
        "domain": [128, 256, 512, 1024],
    },
]