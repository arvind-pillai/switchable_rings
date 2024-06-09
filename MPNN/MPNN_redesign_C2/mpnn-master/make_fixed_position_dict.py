import glob
import random
import numpy as np
import json
import itertools

#MODIFY this path
with open('/home/justas/projects/lab_github/mpnn/data/pdbs.jsonl', 'r') as json_file:
    json_list = list(json_file)

my_dict = {}
for json_str in json_list:
    result = json.loads(json_str)
    all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain']
    fixed_position_dict = {}
    print(result['name'])
    #FIX ONLY PDB NAMED 5TTA in the chain A
    if result['name'] == '5TTA':
        for chain in all_chain_list:
            if chain == 'A':
                fixed_position_dict[chain] = [int(item) for item in list(itertools.chain(list(np.arange(1,4)), list(np.arange(7,10)), [22, 25, 33]))]
            else:
                fixed_position_dict[chain] = []
    else:
        for chain in all_chain_list:
            fixed_position_dict[chain] = []
    my_dict[result['name']] = fixed_position_dict

#MODIFY this path   
with open('/home/justas/projects/lab_github/mpnn/data/pdbs_fixed.jsonl', 'w') as f:
    f.write(json.dumps(my_dict) + '\n')


print('Finished')
#e.g. output
#{"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}, "3LIS": {"A": [], "B": []}}

