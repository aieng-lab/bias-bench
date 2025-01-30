import json

import pandas as pd


# CrowS
df = pd.read_csv("data/crows/crows_pairs_anonymized.csv")
print("Number of samples for CrowS:", len(df))

# SEAT
seats = ['sent-weat6', 'sent-weat6b', 'sent-weat7', 'sent-weat7b', 'sent-weat8', 'sent-weat8b']
n_targets = 0
n_attr = 0
for seat in seats:
    data = json.load(open(f"data/seat/{seat}.jsonl", 'r'))
    n_targets += len(data['targ1']['examples'])
    n_attr += len(data['attr1']['examples'])

print(f"Number of samples for SEAT:")
print(f" - Targets: {n_targets}")
print(f" - Attributes: {n_attr}")



