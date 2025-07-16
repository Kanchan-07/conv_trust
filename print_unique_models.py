import os
import json
from tabulate import tabulate

metadata_dir = os.path.join(os.path.dirname(__file__), 'data', 'metadata')
model_names = set()

for fname in os.listdir(metadata_dir):
    if fname.endswith('.json'):
        fpath = os.path.join(metadata_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model = data.get('agent_model')
                if model:
                    model_names.add(model)
        except Exception as e:
            print(f'Error reading {fname}: {e}')

# Print as a table
print(tabulate([[m] for m in sorted(model_names)], headers=["Unique Model Names"]))
