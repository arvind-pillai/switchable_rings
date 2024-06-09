#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:39:40 2022

@author: apillai1
"""

import os
import sys
import json
import argparse
import numpy as np
import itertools
import random
from pyrosetta import *
from pyrosetta.rosetta import *

from pyrosetta.distributed.packed_pose.core import PackedPose
from pyrosetta.distributed import requires_init
from typing import *
import matplotlib.pyplot as plt
from pyrosetta.rosetta.protocols.grafting import *

def range_CA_align(pose_a, pose_b, start_a, end_a, start_b, end_b):
        """
        Align poses by superimposition of CA given two ranges of indices.
        (pose 1 mobile)
        Modified from apmoyer.
        """
        import pyrosetta

        pose_a_residue_selection = range(start_a, end_a)
        pose_b_residue_selection = range(start_b, end_b)

        assert len(pose_a_residue_selection) == len(pose_b_residue_selection)

        pose_a_coordinates = (
            pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
        )
        pose_b_coordinates = (
            pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
        )

        for pose_a_residue_index, pose_b_residue_index in zip(
            pose_a_residue_selection, pose_b_residue_selection
        ):
            pose_a_coordinates.append(pose_a.residues[pose_a_residue_index].xyz("CA"))
            pose_b_coordinates.append(pose_b.residues[pose_b_residue_index].xyz("CA"))

        rotation_matrix = pyrosetta.rosetta.numeric.xyzMatrix_double_t()
        pose_a_center = pyrosetta.rosetta.numeric.xyzVector_double_t()
        pose_b_center = pyrosetta.rosetta.numeric.xyzVector_double_t()

        pyrosetta.rosetta.protocols.toolbox.superposition_transform(
            pose_a_coordinates,
            pose_b_coordinates,
            rotation_matrix,
            pose_a_center,
            pose_b_center,
        )

        pyrosetta.rosetta.protocols.toolbox.apply_superposition_transform(
            pose_a, rotation_matrix, pose_a_center, pose_b_center
        )
    #    print ("Reached here")
        return

def get_rmsd(design: Pose, prediction: Pose) -> float:
        """Calculate Ca-RMSD of prediction to design"""
        rmsd_calc = pyrosetta.rosetta.core.simple_metrics.metrics.RMSDMetric()
        rmsd_calc.set_rmsd_type(pyrosetta.rosetta.core.scoring.rmsd_atoms(3))
        rmsd_calc.set_run_superimpose(True)
        rmsd_calc.set_comparison_pose(design)
        rmsd = float(rmsd_calc.calculate(prediction))
        return rmsd

  
    
init("-beta -mute all")
count = 0
for filename in os.listdir("/home/apillai1/switchable_ring_pipeline/plddt_filtered/"):
  if "_unrelaxed.pdb" in filename:
    A_alphafold = pose_from_pdb("/home/apillai1/switchable_ring_pipeline/plddt_filtered/"+filename)
    A_alphafold1 = A_alphafold.split_by_chain(1)
    k = filename.split(".fa")
    alt_file = k[0]+".pdb"
    asym = pose_from_pdb("/home/apillai1/switchable_ring_pipeline/X_state_prediction/"+alt_file)
    A = asym#.split_by_chain(1)
    print (A_alphafold.sequence())
    print (A.sequence())
    A_rmsd = get_rmsd(A,A_alphafold1)
    print (A_rmsd)

    if A_rmsd <2.75:
         A_alphafold1 = A_alphafold.split_by_chain(1)
         A_alphafold1.dump_pdb("/home/apillai1/switchable_ring_pipeline/rmsd_filtered/"+filename)
#     print ("A","B",A_rmsd,B_rmsd,str(count))
#    count+=1

