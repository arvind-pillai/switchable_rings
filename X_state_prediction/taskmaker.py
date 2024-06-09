
import os

outfile = open("filenames.txt","w")

for filename in os.listdir("/home/apillai1/switchable_ring_pipeline/combined/"):

    if 'asym' in filename:
        outfile.write("/software/containers/pyrosetta.sif /home/apillai1/switchable_ring_pipeline/05_Make_X_state_monomer.py /home/apillai1/switchable_ring_pipeline/combined/"+filename+'\n')
outfile.close()
      
