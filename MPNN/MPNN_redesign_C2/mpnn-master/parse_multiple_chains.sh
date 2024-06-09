#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=parse_multiple_chains.out

source activate SE3

# Parse a folder with .pdb structures to a dictionary (jsonl) file. 

# Provide a path to your forlder with pdbs to be parsed, 
# Provide a path where to output parsed pdbs into a jsonl file.

python parse_multiple_chains.py --pdb_folder='/projects/ml/struc2seq/combo_pdbs/' --out_path='/projects/ml/struc2seq/multi_chain_test_output/combo_pdbs.jsonl'

