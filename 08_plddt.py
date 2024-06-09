import os
import shutil
infile = open("//home/apillai1/switchable_ring_pipeline/predictions/reports.txt","r")
d = {}
count = 0
for line in infile:
  count+=1

  k = line.split()

  filename = k[0]+"_model_4_ptm_seed_0_unrelaxed.pdb"
  plddt_tag = (k[4])
  r = plddt_tag.split(":")

  plddt = float(r[1])
  d[filename] = plddt
  print (filename,plddt) 
for filename in os.listdir("/home/apillai1/switchable_ring_pipeline/predictions/"):
 if "pdb" in filename and not "json" in filename and filename in d:
  if d[filename] >86:
   print (d[filename])
   shutil.copyfile("/home/apillai1/switchable_ring_pipeline/predictions/"+filename, "/home/apillai1/switchable_ring_pipeline/plddt_filtered/"+filename)

  
#for filename in os.listdir("/home/final_rings_set/alphafold/filtered"):
  
  
