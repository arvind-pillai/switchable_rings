#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:50:13 2022

@author: apillai1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:06:47 2021

@author: apillai1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 11:28:22 2021

@author: apillai1
"""

# special packages on the DIGS

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
    

filename = sys.argv[1] #<-- input scaffold
init("-beta -mute all")

pose = pose_from_pdb(filename)

hinge_asym = pose.split_by_chain(1)#.split_by_chain(1)
hinge_asym2 = pose.split_by_chain(1)#.split_by_chain(1)

if "JHB7Yr2" in filename:

  hinge = pose_from_pdb("/home/apillai1/working/JHB7Xr2.pdb")
  hinge_N = "LRAEKT"
  hinge_C = "EIVKLA"

if "FF74Y" in filename:

  hinge = pose_from_pdb("/home/apillai1/working/FF74X.pdb")
  hinge_N = "VFLAAKASEN"
  hinge_C = "TEGALEA"

if "cs_221" in filename:

  hinge = pose_from_pdb("/home/apillai1/working/cs_221X.pdb")
  hinge_N = "AAAAARS"
  hinge_C = "GSPEEKLEI"
print (hinge_N,hinge_C)
seq_asym = hinge_asym.sequence()
seq_hinge = hinge.sequence()

#align N-term hinge_Y to asym_unit
start_an = seq_hinge.index(hinge_N)+1-10
start_bn = seq_asym.index(hinge_N)+1-10
stop_an = start_an + len(hinge_N)+10
stop_bn = start_bn + len(hinge_N)+10
#range_CA_align(hinge, hinge_asym, start_an, stop_an, start_bn, stop_bn)
range_CA_align(hinge, hinge_asym, start_an, stop_an, start_bn, stop_bn)


print (start_an,start_bn,stop_an,stop_bn)
#align asym_unit to C-term of Hinge_Y
start_ac = seq_hinge.index(hinge_C)+1
start_bc = seq_asym.index(hinge_C)+1
stop_ac = start_ac + len(hinge_C)+10
stop_bc = start_bc + len(hinge_C)+10
#range_CA_align(hinge_asym2, hinge, start_bc, stop_bc, start_ac, stop_ac)
range_CA_align(hinge_asym2, hinge, start_bc, stop_bc, start_ac, stop_ac)

print (start_ac,start_bc,stop_ac,stop_bc)
newpose = Pose()
for i in range (1,stop_bn):
  newpose.append_residue_by_bond(hinge_asym.residue(i))
if "JHB" in filename: #start_bc-stop_bn >1:  
  for i in range (stop_an,start_ac):
    newpose.append_residue_by_bond(hinge.residue(i))
hinge_bottom = len(hinge_asym2.sequence())

for i in range (start_bc,hinge_bottom+1):
  print (i, hinge_bottom)
  newpose.append_residue_by_bond(hinge_asym2.residue(i))

   # homodimer.dump_pdb(filename+"_HOMODIMER.pdb")

k = filename.split("/")
print (k[-1])

newpose.dump_pdb(k[-1])
   # homodimer.dump_pdb(filename+"_HOMODIMER.pdb")   
