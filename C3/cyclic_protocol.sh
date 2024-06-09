#!/bin/bash

sbatch -p short -N 1 -n 1  -J C2_job_two --output C2_job_3_one.log --err C2_job_one.err --mem 5G  --wrap="OMP_NUM_THREADS=1 PYTHONPATH=PYTHONPATH=/software/worms/worms/ /home/apillai1/final_production_run/worms_conda/worms/bin/python -m worms @t32_oneDHR.flags" OMP_NUM_THREADS=1 
