import json
import numpy as np

with open("dataset_noise_test/features_labels.jsonl") as f:
    line = json.loads(f.readline())
    
# Or let's use the env.py logic
