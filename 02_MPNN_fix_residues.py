#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:23:56 2022

@author: apillai1
"""

import sys
import os

directory = "./combined/"  #<-- Folder containing worms outputs selected
outfile = open("fixed_pos.jsonl","w") #<-- List of residues to fix for MPNN
outfile.write("{")
resi_chain = []
count_file = 0
for filename in os.listdir(directory):
  print (filename)
  if "_asym" in filename:
   
  #Fixes these residues to try and preserve hinge conformational switch (shown in EFig 1)
  
    if "JHB7Y" in filename:
      fix_resi = [65,68,69,72,73,75,76,78,79,80,81,84,88,91,95,98,106,107,109,110,113,114,117,120,121,124,127,128,130,131,132,133,134,135,136,137,138,141,144,145,148,149,151,152,155,156,158,159]

    
    if "221Y" in filename:
      fix_resi = [28,59,62, 63, 66, 67, 69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 40, 41, 44, 47, 48, 51, 55, 56, 57, 60, 63, 64, 67, 68, 70, 71, 74, 75, 78, 81, 84, 85, 87, 88, 91, 92, 94, 95, 97, 98, 103, 107,78, 79, 83, 84, 86, 87, 90, 91, 100,  103, 104, 107, 108, 110, 111, 114, 115]



    if "FF74Y" in filename:
      fix_resi = [66,69,70,73,76,80,83,84,87,88,89,92,95,96,99,102,103,106,109,110,113,115,116,117,118,119,121,122,124,125,126,128,129,131,132,135,136,138,139,142,143,146,147,149,151,152,155,158,159,162,165,166,169,173,35,38,39,40,41,43,44,45,46,47,48,50,51,52,53,62,66,69,72,73,76]  

## Reads through each worms asymmetric unit PDB output and uses the row labelled "Modified residues" which records residue positions altered by worms fusion, and isolates those for design. 
## Generates a list of residues to fix, which is supplied to protein MPNN
## Also keeps the hinge residues key to function static

    chains_index = []
    count_file +=1
    if count_file>1:
      outfile.write(",")
    print (count_file,directory+filename)
    infile = open(directory+filename,"r")
    count = 0
    previous_chain = "A"
    previous_resi = "0"
    resi_chain = []
    positions = []
    
    ##Converts PDB residue numbering to index format compatible with ProteinMPNN 
    for line in infile:
      if "ATOM" in line:
        
        k = line.split()
        if len(k[4]) > 4: 
            comb_chain_resi = k[4]
            chain = comb_chain_resi[0]
            resi = comb_chain_resi[1:]
            
            if chain == previous_chain and resi != previous_resi:
               count +=1
              # print (count)
               previous_resi = resi
               resi_chain.append(count)
               chains_index.append(chain)
            if chain != previous_chain:
               count = 1
               previous_chain = chain
               previous_resi = resi
               resi_chain.append(count)  
               chains_index.append(chain)
        else:     
            
            if k[4] == previous_chain and k[5] != previous_resi:
               count +=1
           #    print (count)
               previous_resi = k[5]
               chains_index.append(k[4])
               resi_chain.append(count)
            if k[4] != previous_chain:
               count = 1
               previous_chain = k[4]
               previous_resi = k[5]
               chains_index.append(k[4])
               resi_chain.append(count)           

      #Finds residues around the newly formed junction       
      if "Modified positions" in line:
        line = line.strip()
        k = line.split()
        positions = k[2].split(",")
        
      #Finds hinge positions in chimeric protein 
      if "Segment:  1 resis" in line:
         k = line.split()  
         start_worms = k[3]
         start_worms = int(start_worms.replace("-",""))
         stop_worms = int(k[4])
         dash = k[8].split("-")
         start_hinge = int(dash[0])
         stop_hinge = int(dash[1])
         count = start_hinge
         print (start_worms,stop_worms,start_hinge,stop_hinge)
         hinge_fix_list = []
         for i in range (start_worms,stop_worms):
              if count in fix_resi:
                hinge_fix_list.append(str(i))   
              count +=1
    fixed = []
    chain_index2 = []

   #Generates the list for a given PDB
    for i in range (0,len(resi_chain)):
      if not str(resi_chain[i]) in positions or (str(resi_chain[i]) in hinge_fix_list):
        fixed.append(resi_chain[i])
        chain_index2.append(chains_index[i])

    print (len(fixed),len(chain_index2))
    string = "ABCDEFGHIJK"


    name = filename[:len(filename)-4]

#Deals with formatting for the residue list
    outfile.write('''  "'''+name+'''": ''')
    chains = {}
    for chain in string:
     if chain in chain_index2:   
      resi_list = []
      for i in range (len(fixed)):
        if chain_index2[i] == chain:
          resi_list.append(fixed[i])
      chains[chain] = resi_list
    chains = str(chains)
    chains = chains.replace("'",'''"''')
    outfile.write(chains+"")
outfile.write("}")
outfile.close()


      
