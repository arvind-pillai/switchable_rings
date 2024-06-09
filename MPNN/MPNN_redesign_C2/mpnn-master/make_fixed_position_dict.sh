#!/bin/bash
#SBATCH --mem=32g
#SBATCH -c 2
#SBATCH --output=make_fixed_position_dict.out

source activate SE3


python make_fixed_position_dict.py

