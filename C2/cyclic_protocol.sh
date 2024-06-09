#!/bin/bash

sbatch -p short -N 1 -n 1  -J C2_job_two --output C2_job_3_one.log --err C2_job_one.err --mem 5G  --wrap="OMP_NUM_THREADS=1 PYTHONPATH=/home/thuddy/th_home/worms_backup/worms_beta /home/thuddy/th_home/worms_backup/conda_env/python -m worms @t32_oneDHR.flags"
