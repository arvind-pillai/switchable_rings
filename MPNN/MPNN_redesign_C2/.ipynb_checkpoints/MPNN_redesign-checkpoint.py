#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:23:56 2022

@author: apillai1
"""

#Modified positions: 103,104,107,111,114,114,116,117,117,118,118,120,121,121,122,122,124,126,126,129,129,130,132,132,133,133,136,136,137,140,141,146,147,149,150,150,151,152,153,153,154,154,155,155,156,156,157,157,158,158,159,159,160,160,161,161,162,163,163,164,167,169,170,170,173,173,174,174,176,177,177,178,180,180,181,181,183,184,184,187,188,188,191,199,200,203,203,206,206,207,207,210,210,211,211,213,213,214,214,217

import sys
import os

directory = "/home/apillai1/final_rings_set/combined/"  #sys.argv[1]
outfile = open("fixed_pos.jsonl","w")
outfile.write("{")
resi_chain = []
count_file = 0
for filename in os.listdir(directory):
  
  if "_asym" in filename:# and  "FF74X" in filename and not ("FF63" in filename):
  
    if "JHB7Y" in filename:
      fix_resi = [65,68,69,72,73,75,76,78,79,80,81,84,88,91,95,98,106,107,109,110,113,114,117,120,121,124,127,128,130,131,132,133,134,135,136,137,138,141,144,145,148,149,151,152,155,156,158,159]
   # for i in range (88,146):
   #   fix_resi.append(i)
    
    if "221Y" in filename:
      fix_resi = [28,59,62, 63, 66, 67, 69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 40, 41, 44, 47, 48, 51, 55, 56, 57, 60, 63, 64, 67, 68, 70, 71, 74, 75, 78, 81, 84, 85, 87, 88, 91, 92, 94, 95, 97, 98, 103, 107,78, 79, 83, 84, 86, 87, 90, 91, 100,  103, 104, 107, 108, 110, 111, 114, 115]
   # for i in range (53,93):
   #   fix_resi.append(i)


    if "FF74Y" in filename:
      fix_resi = [66,69,70,73,76,80,83,84,87,88,89,92,95,96,99,102,103,106,109,110,113,115,116,117,118,119,121,122,124,125,126,128,129,131,132,135,136,138,139,142,143,146,147,149,151,152,155,158,159,162,165,166,169,173,35,38,39,40,41,43,44,45,46,47,48,50,51,52,53,62,66,69,72,73,76]  
   # for i in range (71,161):
   #   fix_resi.append(i)


    chains_index = []
    count_file +=1
    if count_file>1:
      outfile.write(",")
    print (count_file,directory+filename)
  #  print (directory+filename)
    infile = open(directory+filename,"r")
    count = 0
    previous_chain = "A"
    previous_resi = "0"
    resi_chain = []
    positions = []
    for line in infile:
     # print(line)
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
  #         count = 1
  #         resi_chain.append(count)
 #       print (k[4],count)
            
      if "Modified positions" in line:
        line = line.strip()
        k = line.split()
     #   print (k)
        positions = k[2].split(",")

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



  #  print (positions)

  #  print (chains_index)
    for i in range (0,len(resi_chain)):
      if not str(resi_chain[i]) in positions or (str(resi_chain[i]) in hinge_fix_list):
        fixed.append(resi_chain[i])
        chain_index2.append(chains_index[i])
  #  print (fixed)
  #  print (chain_index2)
    print (len(fixed),len(chain_index2))
    string = "ABCDEFGHIJK"




  # {"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}, "3LIS": {"A": [], "B": []}} 
    name = filename[:len(filename)-4]
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
    
            #print (fixed)
{"5TTA": {"A": [1, 2, 3, 7, 8, 9, 22, 25, 33], "B": []}, "3LIS": {"A": [], "B": []}}    
 #   for line in infile:
      
