#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=make_tied_positions_dict.out

source activate SE3


python make_tied_positions_dict.py

