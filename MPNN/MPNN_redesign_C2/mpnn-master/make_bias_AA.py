import numpy as np
import json

my_dict = {"A": -0.01, "G": 0.02} #0.1 is a good value to start with

with open('/home/justas/projects/lab_github/mpnn/data/bias_AA.jsonl', 'w') as f:
    f.write(json.dumps(my_dict) + '\n')

#e.g. output
#{"A": -0.01, "G": 0.02}
