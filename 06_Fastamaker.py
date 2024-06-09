import os
outfile = open("alphafold_library.fasta","w")
directory = "/home/apillai1/switchable_ring_pipeline/MPNN_outputs/alignments/" #<-- Folder of MPNN outputs

count_files = 0
accepted = 0
for filename in os.listdir(directory):
  
  print (directory+filename)
  infile = open(directory+filename,"r")
  count_files+=1
  
  count = 0
  for line in infile:
   
  # if not "Native" in line:
    
    if ">" in line:
      name = ">"+filename+"_"+str(count) #.strip()
    if not ">" in line:
      seq = line.strip() #.strip()
      chains = seq.split("/")
      seq = chains[0]
      if not "Native" in name and len(seq)<500:
        outfile.write(name+"\n")
        outfile.write(seq+"\n")
        print ("fasta/"+filename)
        file_rename = filename.replace(".pdb",".pdxb")
        outfile2 = open("fasta/"+file_rename[0:len(file_rename)-3]+"_"+str(count)+".fa","w")
        outfile2.write(name+"\n")
        outfile2.write(seq+"\n")
        outfile2.close()
        count +=1
        accepted+=1/8
    if count>9:
      break
  print (count_files,accepted)
outfile.close()


outfile = open("filenames_fasta.txt","w")
directory = "fasta/"

count_files = 0
for filename in os.listdir(directory):

        outfile.write('/software/lab/superfold/superfold '+directory+ filename +' --models 4 --output_summary --out_dir /home/apillai1/switchable_ring_pipeline/predictions/' +'\n')
        

outfile.close()
      
