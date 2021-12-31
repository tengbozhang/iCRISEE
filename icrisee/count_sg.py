from pathlib import Path
import pandas as pd
import csv
from collections import OrderedDict
import numpy as np
import sys
import argparse
import os
import re
import plotly.express as px
#import Levenshtein





class  count_spacers:
    def __init__(self,prefix,input_file, fastq_file, output_file, gRNA_len):
        print("start to count spacer")
        self.prefix = prefix
        self.input_file = f"{input_file}"
        print("input file:'{}'".format(self.input_file))
        self.fastq_file = fastq_file
        print(self.fastq_file)
        self.output_file = f"/tmp/{prefix}/02map/{output_file}.tsv"
        self.gRNA_len = gRNA_len
        self.num_reads = 0
        self.perfect_matches = 0
        self.non_perfect_matches = 0
        self.p1_base_reverse = 'CACC[ATCG]([ATCG]{' + str(self.gRNA_len) +'})'
        self.p2_base_reverse = '([ATCG]{' + str(self.gRNA_len)+'})[ATCG]GGTG'
        self.run()
    def fq(self):
            fastq = open(self.fastq_file, 'r')
            with fastq as f:
                    while True:
                            l1 = f.readline().rstrip('\n')
                            if not l1:
                                    break
                            l2 = f.readline().rstrip('\n')
                            l3 = f.readline().rstrip('\n')
                            l4 = f.readline().rstrip('\n')
                            yield [l1, l2, l3, l4]
       
    def run(self): 
        if "\t" in open(self.input_file).read():
            self.lib = pd.read_csv(self.input_file,sep="\t", names = ["sgRNA","sequence","Gene"])
        else:
            self.lib = pd.read_csv(self.input_file,sep=",", names = ["sgRNA","sequence","Gene"])
        self.lib = self.lib[['sgRNA', 'Gene', 'sequence']]
        self.dictionary = {seq:0 for seq in self.lib["sequence"]}
      
    # open fastq file
        try:
            handle = self.fq()
        except:
            print("could not find fastq file {}".format(self.fastq_file))
            return

    # process reads in fastq file
#    readiter = SeqIO.parse(handle, "fastq")
#        sequ = self.fq()
    #for sequ in readiter: #contains the seq and Qscore etc.
        for  sequ in self.fq():
            sequ = sequ[1]
        #sequ = str(sequ.seq)
            ref = re.search(self.p1_base_reverse,sequ)
            self.num_reads += 1
        
            if ref:
                guide = ref.group(1)
           # print(guide)
                if guide in self.dictionary:
                    self.dictionary[guide] += 1
                    self.perfect_matches += 1
                elif guide[1:] in self.dictionary:
                    self.dictionary[guide[1:]] += 1
                    self.perfect_matches += 1
                else:
                #print(sequ)
                #print(guide)
                    self.non_perfect_matches += 1
            else:
                reb = re.search(self.p2_base_reverse,sequ)
                if reb:
                    guide = reb.group(1)
                    guide = "".join([{"A":"T","T":"A","C":"G","G":"C","N":"N"}[i] for i in guide][::-1])
                #print(guide)
                    if guide in self.dictionary:
                        self.dictionary[guide] += 1
                        self.perfect_matches += 1
                    elif guide[1:] in self.dictionary:
                        self.dictionary[guide[1:]] += 1
                        self.perfect_matches += 1                
                    else:
                   # print(sequ)
                    #print(guide)
                        self.non_perfect_matches += 1        
        with open(self.output_file,'w') as ww:
            for seq in self.lib.sequence:
                ww.write("{}\n".format(self.dictionary[seq]))
        self.key_not_found = self.num_reads-self.perfect_matches-self.non_perfect_matches
        print(self.perfect_matches,self.non_perfect_matches,self.key_not_found)
        return self.perfect_matches,self.non_perfect_matches,self.key_not_found
class write_count:
    def __init__(self,prefix,libfile,countlist,outfile="analysis/01fastq/count_table.csv"):
        self.prefix = prefix
        if "\t" in open(libfile).read():
            self.lib = pd.read_csv(libfile,sep = "\t",names = ["sgRNA","sequence","Gene"])
        else:
            self.lib = pd.read_csv(libfile,sep = ",",names = ["sgRNA","sequence","Gene"])
        self.lib = self.lib[["sgRNA","Gene"]]
        for sample in countlist:
            self.lib[sample]  = pd.read_csv(f"/tmp/{self.prefix}/02map/{sample}.tsv",sep = "\t",header=None)[[0]]
            self.lib.to_csv(outfile,index = None,sep = "\t")


class Fastq:
    def __init__(self,fastqfile,samplelist,prefix,lib="lib.csv",gRNA_len=20):
        self.lib = lib
        self.prefix = prefix
        self.fastqfile = fastqfile
        self.samplelist = samplelist
        self.gRNA_len = gRNA_len
#        self.count = "analysis/01fastq/{}.csv".format(prefix)
        self.count = f"/tmp/{prefix}/count.tsv"
        self.run()
    def run(self):
        maplist = []
        for i in range(len(self.fastqfile)):
            print(self.prefix)
            print(self.lib)
            print(self.fastqfile[i])
            print(self.samplelist[i])
            print(self.gRNA_len)
            counts = count_spacers(self.prefix,self.lib, self.fastqfile[i], self.samplelist[i], self.gRNA_len)
            maplist.append(counts.run())
        mapped = pd.DataFrame(maplist)
        mapped.columns = ["Perfect","Unperfect","Nokey"]
        mapped = mapped[["Perfect","Nokey","Unperfect"]]
        mapped["sample"] = self.samplelist
        pietu = mapped.T.iloc[:3,:]
        pietu.columns=mapped["sample"]
        pietu.to_csv(f"/tmp/{self.prefix}/map.txt",sep="\t")
        write_count(self.prefix,self.lib,self.samplelist,self.count)
########function completed

if __name__ == "__main__":

    fastqfile = ["MZcTEQP9TD_1.fastq","MZcTEQP9TD_2.fastq","MZcTEQP9TD_3.fastq","MZcTEQP9TD_4.fastq"]
    samplelist = ["ctr1","ctr2","treat1","treat2"]
     
    fastqfile = ["5vLSEscGa3Su9WlVxJ6G_1.fastq","5vLSEscGa3Su9WlVxJ6G_2.fastq","5vLSEscGa3Su9WlVxJ6G_3.fastq","5vLSEscGa3Su9WlVxJ6G_4.fastq"]
    samplelist = ["control1","control2","treat1","treat2"]
    count_spacers("5vLSEscGa3Su9WlVxJ6G","5vLSEscGa3Su9WlVxJ6G/5vLSEscGa3Su9WlVxJ6G.csv","5vLSEscGa3Su9WlVxJ6G_1.fastq",'control1',20)
    Fastq(fastqfile,samplelist,"5vLSEscGa3Su9WlVxJ6G","5vLSEscGa3Su9WlVxJ6G.csv",20)


