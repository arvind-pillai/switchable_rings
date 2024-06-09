
import pyrosetta
import os
import glob

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

def distance (a,b):
  return ((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)**0.5


def range_CA_align(pose1, pose2, start1, end1, start2, end2):
#modified from Adam's function (pose 1 mobile)
    if end1 <0: 
        end1 = pose1.size()+1+end1
    if end2 == 'auto':
        end2 = start2 + end1 - start1
    elif end2 <0: 
        end2 = pose2.size()+2+end2
    if start2 == 'auto':
        start2 = start1 -end1 + end2
    
    print(start1, end1, start2, end2)
    pose1_residue_selection = range(start1,end1)
    pose2_residue_selection = range(start2,end2)
    
    assert len(pose1_residue_selection) == len(pose2_residue_selection)

    pose1_coordinates = pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()
    pose2_coordinates = pyrosetta.rosetta.utility.vector1_numeric_xyzVector_double_t()

    for pose1_residue_index, pose2_residue_index in zip(pose1_residue_selection, pose2_residue_selection):
        pose1_coordinates.append(pose1.residues[pose1_residue_index].xyz('CA'))
        pose2_coordinates.append(pose2.residues[pose2_residue_index].xyz('CA'))

    rotation_matrix = pyrosetta.rosetta.numeric.xyzMatrix_double_t()
    pose1_center = pyrosetta.rosetta.numeric.xyzVector_double_t()
    pose2_center = pyrosetta.rosetta.numeric.xyzVector_double_t()

    pyrosetta.rosetta.protocols.toolbox.superposition_transform(pose1_coordinates,
                                                                pose2_coordinates,
                                                                rotation_matrix,
                                                                pose1_center,
                                                                pose2_center)

    pyrosetta.rosetta.protocols.toolbox.apply_superposition_transform(pose1,
                                                                      rotation_matrix,
                                                                      pose1_center,
                                                                      pose2_center)
                     

def clash_check(pose: Pose) -> float:
    """
    Get fa_rep score for a pose with weight 1. Backbone only.
    Mutate all residues to glycine then return the score of the mutated pose.
    """

    import pyrosetta

    # initialize empty sfxn
    sfxn = pyrosetta.rosetta.core.scoring.ScoreFunction()
    sfxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1)
    # make the pose into a backbone without sidechains
    all_gly = pose.clone()
    true_sel = pyrosetta.rosetta.core.select.residue_selector.TrueResidueSelector()
    true_x = true_sel.apply(all_gly)
    # settle the pose
    pyrosetta.rosetta.protocols.toolbox.pose_manipulation.repack_these_residues(
        true_x, all_gly, sfxn, False, "G"
    )
    score = sfxn(all_gly)
    return score
    

pyrosetta.init("-mute all")
sfxn = pyrosetta.rosetta.core.scoring.ScoreFunction()
sfxn.set_weight(pyrosetta.rosetta.core.scoring.fa_rep, 1)
outfile  = open("Atomic_distance.txt","w")
for filename in os.listdir("/home/apillai1/switchable_ring_pipeline/rmsd_filtered_example"): #<-- Change this to relevant directory
 
 if "pdb" in filename and not "json" in filename:
  print ("Current file: ",filename)
  if "LHD289" in filename or "LHD321" in filename or "LHD278" in filename or "LHD202" in filename or "LHD317" in filename:
    if "LHD278" in filename:
      LHD289 = pose_from_pdb("/home/apillai1/LHD278.pdb")      
    if "LHD289" in filename:
      LHD289 = pose_from_pdb("/home/apillai1/LHD289_short.pdb")
    if "LHD321" in filename:
      LHD289 = pose_from_pdb("/home/apillai1/LHD321.pdb")
    if "LHD202" in filename:
      LHD289 = pose_from_pdb("/home/apillai1/Bridge/4_fusion_bridge/DHR_fusions/combinations/202A62B57.pdb")
    if "LHD317" in filename:
      LHD289 = pose_from_pdb("/home/apillai1/Bridge/4_fusion_bridge/af2_validated/sel_rmsd2-5_plddt88/design/LHD317_DHR57_BA_A_0003_0003.pdb")
    k = filename.split("_")
 #   sym = k[0]
    
    if "C5" in filename:
      original_size = 5
    if "C4" in filename:
      original_size = 4
    if "C3" in filename:
      original_size = 3
    if "C2" in filename:
      original_size = 2
        #   sym = int(sym[-1])
    sym = 3
    hinge_connector = pose_from_pdb("/home/apillai1/switchable_ring_pipeline/rmsd_filtered_example/"+filename)
    monomer = hinge_connector.split_by_chain(1)
    monomer2 = hinge_connector.split_by_chain(1)
    sequential_docking = []
    distances = []
    for i in range (sym):
        length_LHD = len(LHD289.sequence())
        LHD289_A = LHD289.split_by_chain(1)
        length_monomer = len(monomer2.sequence())
        length_A = len(LHD289_A.sequence())

        range_CA_align(LHD289, monomer, length_A+1, length_A+50, 1, 50)

        range_CA_align(monomer2, LHD289, length_monomer-50, length_monomer, length_A-50, length_A)
      
        monomer2.append_pose_by_jump(monomer,1)
        start = 1
        stop = len(monomer2.sequence())
        shortest = 10000
        sequential_docking.append(monomer2.clone())
        
        for atom1 in range (1,100):
          CA_x1 = np.array(monomer2.residue(atom1).xyz("CA"))
          for atom2 in range(len(monomer2.sequence())-100,len(monomer2.sequence())):
            CA_x2 = np.array(monomer2.residue(atom2).xyz("CA"))
            if distance(CA_x1,CA_x2)<shortest:
              shortest = distance(CA_x1,CA_x2)
        distances.append(shortest)
        print ("Here:",monomer2.num_chains())
        subunit_no = {0:"Dimer",1:"Trimer",2:"Tetramer"}
        outfile.write(subunit_no[i]+","+filename+","+str(shortest)+"\n")
        
         
        print (filename,shortest)

    shortest_distance = 1000
    num_subunits = 5
      
    for i in range(len(sequential_docking)):
        if distances[i]<shortest_distance:
          shortest_distance = distances[i]
          num_subunits = i
        if distances[i]<24:
          break  
    print ("Smallest number",num_subunits)
    final = sequential_docking[num_subunits]
    final.dump_pdb("X_state_dock/"+filename)
    print (filename)
outfile.close()
