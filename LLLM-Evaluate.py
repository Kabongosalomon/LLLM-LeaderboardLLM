import json
import os
import random

import numpy as np
import torch
from evaluation_metrics import Metrics

seed = 42
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


labels = """
[
    {
        "leaderboard": {
            "Task": "",
            "Dataset": "",
            "Metric": "",
            "Score": "",
        }
    },
    {
        "leaderboard": {
            "Task": "aaaa AAAA",
            "Dataset": "bbb BBB",
            "Metric": "cc CC",
            "Score": "d D",
        }
    },
    {
        "leaderboard": {
            "Task": "e E",
            "Dataset": "ff FF",
            "Metric": "gg GG",
            "Score": "hh HH",
        }
    }     	
]
"""

decoded_preds = """
[
    {
        "leaderboard": {
            "Task": "",
            "Dataset": "",
            "Metric": "",
            "Score": "",
        }
    },
    {
        "leaderboard": {
            "Task": "aaaa AAAA",
            "Dataset": "bbb BBB",
            "Metric": "cc CC",
            "Score": "d D",
        }
    },
    {
        "leaderboard": {
            "Task": "e E",
            "Dataset": "ff FF",
            "Metric": "gg GG",
            "Score": "hh HH",
        }
    }     	
]
"""

result = Metrics.evaluate_property_wise_json_based(label_list=labels, prediction_list=decoded_preds)

result.update(Metrics.evaluate_rouge(label_list=labels, prediction_list=decoded_preds))

print(result)
