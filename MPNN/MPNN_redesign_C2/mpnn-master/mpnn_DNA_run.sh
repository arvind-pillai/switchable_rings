#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=mpnn_DNA_run.out

source activate mlfold
# Default model

#        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10/checkpoints/epoch51_step255000.pt' \
#        --hidden_dim 192 \

# Smaller model for larger proteins if run time, or memory is an issue, also might want to reduct num_connections to 48, or 32

#        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p7/checkpoints/epoch114_step570000.pt' \
#        --hidden_dim 128 \

python mpnn_DNA_run.py \
        --max_length 10000 \
        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10_DNA/checkpoints/epoch93_step300000.pt' \
        --hidden_dim 192 \
        --num_layers 3 \
        --protein_features 'full' \
        --jsonl_path='/home/justas/projects/lab_github/mpnn/DNA_data/4h10.jsonl' \
        --chain_id_jsonl '/home/justas/projects/lab_github/mpnn/DNA_data/4h10_masked.jsonl'  \
        --fixed_positions_jsonl '' \
        --out_folder='/home/justas/projects/lab_github/mpnn/DNA_data/' \
        --num_seq_per_target 32 \
        --sampling_temp="0.1" \
        --batch_size 8 \
        --omit_AAs 'Xatcgdryu' \
        --backbone_noise 0.05 \
	--bias_AA_jsonl '' \
        --num_connections 64 \
        --omit_AA_jsonl ''

