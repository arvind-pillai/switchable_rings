#!/bin/bash
#SBATCH -p cpu
#SBATCH -t 00:05:00
#SBATCH -J C2_sym_D2
#SBATCH --mem=12g
#SBATCH -o out
#SBATCH -a 1-2000
# get line number ${SLURM_ARRAY_TASK_ID} from tasks file

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" filenames.txt)
# tell bash to run $CMD
echo "${CMD}" | bash

