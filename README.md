# De novo design of allosterically switchable protein assembles

[![B1-C3-to-C4.jpg](https://i.postimg.cc/s2mBrcCz/B1-C3-to-C4.jpg)](https://postimg.cc/v11875m2)


The following tools used in Pillai et al. 2024 can be obtained from the GitHub repositories listed below

### Relevant repositores

RF-Diffusion: https://github.com/RosettaCommons/RFdiffusion

Worms: https://github.com/mingkangyang/worms-

ProteinMPNN: https://github.com/dauparas/ProteinMPNN

Superfold: https://github.com/rdkibler/superfold

### Installing and running pipelines

This pipeline generates hinge-containing rings whose interfaces are derived from LHDs. These scripts were written to be run on the Institute of Protein Design's DIGs server, however, minor modifications of path filenames (denoted by comments in each script) should allow these scripts to be adapted for other platforms.

Install apptainer so that you can run this code without installing python packages, but rather with a prepackaged sif file: https://apptainer.org/docs/admin/main/installation.html

Download the following .sif files and zip files for worms and worms_conda 

```wget http://files.ipd.uw.edu/pub/switchable_rings/pyrosetta.sif```

```wget http://files.ipd.uw.edu/pub/switchable_rings/mlfold.sif```

```wget http://files.ipd.uw.edu/pub/switchable_rings/worms.zip```

```wget http://files.ipd.uw.edu/pub/switchable_rings/worms_conda.zip```


Download worms and worms_conda:
wget 

This pipeline generates hinge-containing rings whose interfaces are derived from LHDs. The steps are listed below


1) WORMs based generation of Y-state ring scaffolds, using building blocks shown in supplementary materials in /building_blocks/hinges_used (for cs221Y, JHB7Y and FF74Y) and /building_blocks/LHD_DHRs for LHD-DHR fusions. Example C2, C3 and C4 ring scaffolds are deposited in separate folders with those respective names. THe FLAGs provided to WORMS are shown in the file: t32_oneDHR.flags, while the server submission request is shown in cyclic_protocol.sh for each. The WORMS database used, along with filenames and splicing conditions used is deposited in /worms_database as a json file. Worms is run with the following command:
```OMP_NUM_THREADS=1 PYTHONPATH=/path/to/worms/ /path/to/worms_conda/worms/bin/python -m worms @ring_oneDHR.flags```

2) ```/path/to/pyrosetta.sif 01_filter_length.py``` isolates scaffolds from each worms output folder that meet a given length threshhold (<450 aa) and labels them by symmetry, depositing them in /combined

3) ```/path/to/pyrosetta.sif 02_MPNN_fix_residues.py``` identifies a subset of fusion-modified junction residues for redesign, while fixing key hinge residues, generating a file called fixed_pos.jsonl that includes residues that should remain unchanged during MPNN redesign.

4) Use the jupyter notebook, 03_MPNN_loading_example.ipynb to generate additional files required for MPNN, including pdbs_test.jsonl and pdbs_masked.jsonl. Use shifty.sif as environment.

5) ```sbatch 04_MPNN.sh``` submits an MPNN job to the DIGs server, depositing redesigned sequences in /MPNN_outputs/alignments

6) ```/path/to/pyrosetta.sif /X_state_prediction/05_Make_X_state_monomer.py``` generates the expected X-state for any given Y-state chimera generated in Step 1 by sub-alignment. To run this in batch on the files generated in /combined, use /X_state_prediction/taskmaker.py to generate a list of files to operate on, and sbatch /X_state_prediction/submit.sh to submit this task file as a batch submission, with outputs being generated in the same folder. 

7) ```/path/to/pyrosetta.sif 06_Fastamaker.py``` generates individual fasta files for MPNN-designed sequences in /fasta. 

8) ```/path/to/pyrosetta.sif 07_alphafold_taskmaker.py``` generates a task file and batch submission script for submitting the designed sequences in /fasta to alphafold, using superfold as a wrapper (https://github.com/rdkibler/superfold). Outputs are deposited in /predictions along with estimated plddt in reports.txt.

9) ```/path/to/pyrosetta.sif 08_plddt.py``` filters for designs with an AF2 plddt above a threshhold (default >85), depositing them in /plddt_filtered

10) ```/path/to/pyrosetta.sif 09_rmsd_filter.py``` for designs with an alphafold prediction below a certain rmsd to their X-state predicted monomer (default <2.5A), depositing them in /rmsd_filtered

11) ```/path/to/pyrosetta.sif 10_docker.py``` generates X_state docks out of the alphafold predictions, depositing the alternative ring with the smallest approach of the terminal LHD domains. It also prints distances of closest approach for all of the possible docks in Atomic_distances.txt.

For the inducible dimers, instead of step (11), we ran the detect_clashing_monomers.py script in /C2_clashing to isolate monomers that are expected to clash in the X state - some example outputs are shown there.
