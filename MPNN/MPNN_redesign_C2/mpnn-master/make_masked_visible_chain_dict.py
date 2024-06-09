import glob
import random

import json

#MODIFY THIS PATH - it is a path to the parsed pdb files
with open('/home/justas/projects/lab_github/mpnn/data/pdbs.jsonl', 'r') as json_file:
    json_list = list(json_file)

my_dict = {}
for json_str in json_list:
    result = json.loads(json_str)
    all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    masked_chain_list = ['A'] #predict sequence of chain A
    visible_chain_list = ['B'] #allow to use chain B as a context
    my_dict[result['name']]= (masked_chain_list, visible_chain_list)

#MODIFY THIS PATH FOR THE OUTPUT DICTIONARY
with open('/home/justas/projects/lab_github/mpnn/data/pdbs_masked.jsonl', 'w') as f:
    f.write(json.dumps(my_dict) + '\n')


print('Finished')
# Output looks like this:
# {"5TTA": [["A"], ["B"]], "3LIS": [["A"], ["B"]]}

