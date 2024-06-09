#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=mpnn_run.out

source activate mlfold

python /home/apillai1/final_rings_set/MPNN/MPNN_redesign_C2/mpnn-master/mpnn_run.py        --max_length 10000         --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/paper_experiments/model_outputs/p10/checkpoints/epoch51_step255000.pt'         --hidden_dim 192         --num_layers 3         --protein_features 'full'         --jsonl_path='/home/apillai1/final_rings_set/MPNN/MPNN_redesign_C2/mpnn-master/pdbs_test.jsonl'         --fixed_positions_jsonl '/home/apillai1/final_rings_set/MPNN/MPNN_redesign_C2/fixed_pos.jsonl'         --out_folder='/home/apillai1/final_rings_set/MPNN/MPNN_redesign_C2/output/'         --num_seq_per_target 8         --sampling_temp="0.1"         --batch_size 4         --backbone_noise 0.01         --decoding_order 'random'         --num_connections 64 	--chain_id_jsonl '/home/apillai1/final_rings_set/MPNN/MPNN_redesign_C2/mpnn-master/pdbs_masked_test.jsonl' --omit_AAs 'X'  

