MPNN code runs on pytorch 1.7> on both CPUs and GPUs, use `source activate mlfold` when running on digs.

To run the code see the notebook for data loading: `data_loading_example.ipynb`

To generate sequences with the latest MPNN use: `mpnn_run.sh`

If there is positional tying/symmetries involved use: `mpnn_run_tied.sh`

If structure does not have O (oxygen) then one can use: `mpnn_run_O.sh` - not recommended, but should work in the same way.

Possible features:
1) Specify which amino acid positions need to be fixed in the chain which is designed.
2) Add global amino acid bias.
3) Add global amino acid restriction.
4) Add per position amino acid restrictions.
5) Tie together positions from the same, or different chains - symmetry mode.
6) Add pssm_log_odds bias, AA restrictions from pssm etc.

Slides:
https://docs.google.com/presentation/d/1CFndUOhmifEgoeFCNIeQBip_TqqFB_6lLlt1bfJR5k0/edit?usp=sharing

To generate SSM predictions run: `mpnn_SSM_run.sh`

