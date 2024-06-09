import sys
import os
import shutil

def PDB_len(a):
  infile3 = open(a,"r")
  l = 0
  for line in infile3:
      if "ATOM" in line:
         k = line.split()
         if float(k[5])>l:
             
             l = float(k[5])
             
  infile3.close()
  return l

folders = ["C2","C3","C4","C5"] #<--Folders containing outputs with different symmetries

for elt in folders:
  for filename in os.listdir(""+elt+"/"):
      if PDB_len(""+elt+"/"+filename) <450:
        print (PDB_len(""+elt+"/"+filename), filename)
        shutil.copyfile(""+elt+"/"+filename,""+"combined"+"/"+elt+"_"+filename)
