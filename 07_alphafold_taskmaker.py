
import os

outfile = open("filenames.txt","w")

for filename in os.listdir("./fasta"): #<--folder containing fasta files
    k = filename.split("asym")
    file = k[0].replace("pdxb","pdb")
    file_guess = file+"asym"+".pdb"
    if '.fa' in filename:
        outfile.write('/software/scripts/superfold/superfold fasta/'+ filename +" --models 1 --output_summary --out_dir predictions/ "+'\n')
outfile.close()
      
