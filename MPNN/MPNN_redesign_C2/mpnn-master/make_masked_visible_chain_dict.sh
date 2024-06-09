#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=make_masked_visible_chain_dict.out

source activate SE3

#Adjust this script to adjust masked/visible chains
#The output will be a dictionary with entries {"3LVK_combo": [["A", "C"], ["B"]]}, keys giving a protein name, values giving a list of two lists, the first list is a list of masked chains (sequence for those chains need to be predicted), e.g. ["A", "C"], the second list is a list of visible chains (model will use the sequence of that chain together with the backbone to predict other chain AAs), in this case ["B"]

python make_masked_visible_chain_dict.py

