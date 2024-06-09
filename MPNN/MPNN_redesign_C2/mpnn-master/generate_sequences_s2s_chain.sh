#!/bin/bash
#SBATCH -p gpu
#SBATCH --mem=32g
#SBATCH --gres=gpu:rtx2080:1
#SBATCH -c 3
#SBATCH --output=generate_sequences_s2s_chain.out

source activate mlfold
#source activate mlfold #if using a6000 GPU

#--max_length 100  ###adjust this to include larger proteins, the model was trained with max_length=5000, but should work beyond this
#--jsonl_path  ###add a path to your parsed pdbs
#--chain_id_jsonl ###add a path to your chain masks/ids
#--out_folder ###add a path to a folder where the sequences and scores will be written
#--num_seq_per_target=4 ###number of sequences to sample per backbone/pdb
#--sampling_temp=0.2 ###sampling temperature, 0.2 is a good starting point, increase temperature for more diverse sequence outputs (potentially will need to generate more sequence to find a good one), decrease to have less diverse sequence.
#--batch_size=4 ###number of sequences to process at once, decrease this number if running out of GPU memory
#--checkpoint_path ###this is a path to trained model weights, does not need to be changed
#--hidden_dim ### hidden dimension size for the model, fixed!
#--num_layers ### number of encoder/decoder lyers for the model, fixed!
#--backbone_noise ###standard deviation of the random noise added to the input backbone atoms in Angstroms
#--omit_AAs ###a list of amino acids to omit from generating, e.g. 'AGPCW'
#--fixed_positions_jsonl ###a dictionary to add fixed positions for particular chains in PDBs; if '' then it would not be used.
#--bias_AA_jsonl ### a dictionary of biases to add to sequences, {A: 0.2, G: -0.2} 
#Test use:
#--fixed_positions_jsonl '/projects/ml/struc2seq/multi_chain_test_output/combo_pdbs_position_fixed.jsonl' \

#DEFAULT MODEL
#Train loss: 4.780, Validation loss: 4.568
#Random, 256, N, CA, C model:
#       --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/new_frame_v4/checkpoints/epoch35_step175000.pt' \
#       --protein_features 'full' \
#       --decoding_order 'random' \
#       --hidden_dim 256 \
#       --num_connections 64
#       --more_frames 'true'

#Train loss: 5.71, Validation loss: 5.47
#Random, 256, N, CA, C model:
#       --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m3_random_loss_bias_var/checkpoints/epoch45_step557592.pt' \
#       --protein_features 'full' \
#       --decoding_order 'random' \
#       --hidden_dim 256 \
#       --num_connections 64


#Train loss: 5.25, Validation loss: 5.50
#Random, 256, N, CA, C model:
#       --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m3_random/checkpoints/epoch59_step568111.pt' \
#       --protein_features 'full' \
#       --decoding_order 'random'
#       --hidden_dim 256 \


#Train loss: 5.74, Validation loss: 5.81
#Random, 128, N, CA, C model:
#       --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m1_random/checkpoints/epoch52_step500708.pt' \
#       --protein_features 'full' \
#       --decoding_order 'random'
#       --hidden_dim 128 \


#Train loss: 5.60, Validation loss: 5.98
#Random, 256, CA model:
#        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m3_CA_random/checkpoints/epoch46_step442934.pt' \
#        --protein_features 'coarse' \
#        --decoding_order 'random'
#        --hidden_dim 256 \


#Train loss: 6.00, Validation loss: 6.23
#Random, 128, CA model:
#        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m1_CA_random/checkpoints/epoch72_step693288.pt' \
#        --protein_features 'coarse' \
#        --decoding_order 'random'
#        --hidden_dim 128 \



#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------


#Train loss: 5.37, Validation loss: 5.62
#Forward, 256, N, CA, C model:
#       --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m3/checkpoints/epoch23_step221467.pt' \
#       --protein_features 'full' \
#       --decoding_order 'forward'
#       --hidden_dim 256 \


#Train loss: 5.71, Validation loss: 6.07
#Forward, 256, CA model:
#        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m3_CA/checkpoints/epoch20_step192580.pt' \
#        --protein_features 'coarse' \
#        --decoding_order 'forward'
#        --hidden_dim 256 \


#Train loss: 5.61, Validation loss: 5.78
#Forward, 128, N, CA, C smaller model:
#       --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/multi_m1/checkpoints/epoch81_step779949.pt' \
#       --protein_features 'full' \
#       --decoding_order 'forward'
#       --hidden_dim 128 \

 

python generate_sequences_s2s_chain.py \
        --max_length 10000 \
        --checkpoint_path '/projects/ml/struc2seq/data_for_complexes/training_scripts/models/new_frame_v4/checkpoints/epoch35_step175000.pt' \
        --hidden_dim 256 \
        --num_layers 3 \
        --protein_features 'full' \
        --jsonl_path='/home/justas/projects/lab_github/mpnn/data/pdbs.jsonl' \
        --chain_id_jsonl '/home/justas/projects/lab_github/mpnn/data/pdbs_masked.jsonl'  \
        --fixed_positions_jsonl '/home/justas/projects/lab_github/mpnn/data/pdbs_fixed.jsonl' \
        --out_folder='/home/justas/projects/lab_github/mpnn/output_v2' \
        --num_seq_per_target 32 \
        --sampling_temp="0.1" \
        --batch_size 8 \
        --omit_AAs 'X' \
        --backbone_noise 0.05 \
        --decoding_order 'random' \
	--bias_AA_jsonl '' \
        --num_connections 48 \
        --more_frames 'true' \
        --omit_AA_jsonl '/home/justas/projects/lab_github/mpnn/data/omit_AA.jsonl'
