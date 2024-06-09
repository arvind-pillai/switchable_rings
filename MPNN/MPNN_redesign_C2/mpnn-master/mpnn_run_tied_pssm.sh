#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=mpnn_run_tied.out

source activate mlfold
# Default model

#        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10/checkpoints/epoch51_step255000.pt' \
#        --hidden_dim 192 \

# Smaller model for larger proteins if run time, or memory is an issue, also might want to reduct num_connections to 48, or 32

#        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p7/checkpoints/epoch114_step570000.pt' \
#        --hidden_dim 128 \

python /home/justas/projects/lab_github/mpnn/mpnn_run_tied.py \
        --max_length 10000 \
        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10/checkpoints/epoch51_step255000.pt' \
        --hidden_dim 192 \
        --num_layers 3 \
        --protein_features 'full' \
        --jsonl_path='/home/justas/projects/cages/parsed/test.jsonl' \
        --chain_id_jsonl '/home/justas/projects/cages/parsed/test_masks.jsonl' \
        --fixed_positions_jsonl '' \
        --tied_positions_jsonl '/home/justas/projects/cages/parsed/tied.jsonl' \
        --out_folder='/home/justas/projects/cages/outputs' \
        --num_seq_per_target 4 \
        --sampling_temp="0.1" \
        --batch_size 2 \
        --omit_AAs 'X' \
        --backbone_noise 0.05 \
        --decoding_order 'random' \
        --bias_AA_jsonl '' \
        --num_connections 16 \
        --pssm_jsonl '/home/justas/projects/lab_github/mpnn/data/pssm_dict.jsonl' \
        --pssm_multi 0.1 \
        --pssm_threshold 0.0 \
        --pssm_log_odds_flag 1 \
        --pssm_bias_flag 1

