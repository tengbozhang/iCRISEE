import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
#import modin.pandas as pd
import sys
import numpy as np
from scipy.stats import beta
from statsmodels.distributions.empirical_distribution import ECDF
import random
import multiprocessing 
from multiprocessing import Pool
from pathlib import Path
from statsmodels.sandbox.stats.multicomp import multipletests
#from scipy.interpolate import spline
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
#from matplotlib_venn import venn3, venn3_circles
#from matplotlib_venn import venn2, venn2_circles
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#mpl.rcParams['figure.dpi'] = 600

### define how to get rho value from gene rank
### define how to get rho value from gene rank

class Generank:
    def __init__(self,dataset,model,generank,guide=6,controlsg=1,random_num=20000):
        self.J = None
        self.random_num = random_num
        self.dataset = dataset
        self.model = model
        self.guide = guide
        self.generank = generank
        Path(self.dataset).mkdir(parents=True, exist_ok=True)
        Path(self.dataset+"/01Count").mkdir(parents=True, exist_ok=True)
        Path(self.dataset+"/02sgRNA").mkdir(parents=True, exist_ok=True)
        Path(self.dataset+"/03Gene/Top").mkdir(parents=True, exist_ok=True)
        Path(self.dataset+"/04Pathway").mkdir(parents=True, exist_ok=True)
        self.sgrankfile = "{}/02sgRNA/df_{}.csv".format(self.dataset,self.model) 
        self.rotfile = "{}/03Gene/grra_{}.csv".format(self.dataset,self.model)
        self.rrafile = "{}/03Gene/arra_{}.csv".format(self.dataset,self.model)
        self.avgsfile = "{}/03Gene/avgscore_{}.csv".format(self.dataset,self.model)
        self.sigsfile = "{}/03Gene/sigscore_{}.csv".format(self.dataset,self.model)
        self.alpha=0.25
        self.num_cores = multiprocessing.cpu_count()
        if controlsg ==1 :
            self.controlsg = pd.read_csv(self.dataset+"/controlsg.txt",sep="\t",header=None).iloc[:,0].to_list()
        else:
            self.controlsg = []
        self.padj= "fdr_tsbh"
        self.df_rank = pd.read_csv(self.sgrankfile,sep = "\t").fillna('ctr')
        self.df_rank = self.df_rank[["sgRNA","Gene","pvalue","score","fc"]]
        self.df_rank = self.df_rank[["sgRNA","Gene","score","fc"]]

        self.df_rank["pos_neg"] =  np.where(self.df_rank.fc > 1,"pos","neg")
        self.genes = self.df_rank.Gene.unique()
        self.df_rank["nor_rank"] = (self.df_rank.index+1)/np.shape(self.df_rank)[0]
        try:
            self.controlgene = self.df_rank[self.df_rank.sgRNA.isin(self.controlsg)].Gene
        except:
            self.controlgene = []
        self.get_general_sg() 
        self.run()
    def run(self):
        if self.generank=="grra":
            self.run_icrisee()
        elif self.generank=="arra":
        
            self.run_rra()
        elif self.generank=="avgscore":
            self.run_avgscore()
        elif self.generank=="sigscore":
            self.run_sumscore()
        elif self.generank=="all":
            self.run_rra()
            self.run_icrisee()
            self.run_sigscore()
            self.run_sumscore()
        
    def get_r(self,list_index):
        list_index = sorted(list_index)
    
        rho_k = []
        eff_sg = []
        if len(list_index) <= self.J:
            for k in range(len(list_index)):
                kth_smallest = list_index[k]
                rho_k.append(beta.cdf(kth_smallest,k+1,len(list_index)-k))
            return min(rho_k)     
        else:
            eff_sg = list(list_index)[:self.J] 
            for index in list(list_index)[self.J:]:
                if self.df_rank["fc"][index]>1 and self.df_rank["nor_rank"][index]<0.05:
                    eff_sg.append(self.df_rank["nor_rank"][index])
        
            for sgnum in range(self.J,len(eff_sg)+1): 
                for k in range(sgnum):
                    kth_smallest = eff_sg[k]
                    rho_k.append(beta.cdf(kth_smallest,k+1,sgnum-k))
            return min(rho_k)
    def get_r(self,list_index):
        list_index = sorted(list_index)

        rho_k = []
        eff_sg = []
        for index in list_index:
            if self.df_rank["nor_rank"][index]<0.05 and self.df_rank["fc"][index]>1:
                eff_sg.append(self.df_rank["nor_rank"][index])
        if len(eff_sg) ==0:
            return 1

        for k in range(len(eff_sg)):
            kth_smallest = eff_sg[k]
            rho_k.append(beta.cdf(kth_smallest,k+1,len(eff_sg)-k))
        return min(rho_k)/min(self.guide,len(eff_sg))
    ### define how to get all gene's rho vallue from gene rank
    def get_rho(self,list_index):
        list_index = sorted(list_index)
        rho_k = []
        eff_sg = []
        for index in list_index:
            if self.df_rank["nor_rank"][index]<0.05:
                eff_sg.append(self.df_rank["nor_rank"][index]) 
        if len(eff_sg) ==0:
            return 1

        for k in range(len(eff_sg)):
            kth_smallest = eff_sg[k]
            rho_k.append(beta.cdf(kth_smallest,k+1,len(eff_sg)-k))
        return min(rho_k)
    def get_sumscore(self,list_index):
        list_index = sorted(list_index)
        tmp_gene_df = self.df_rank[self.df_rank.index.isin(list_index)]
        avgs = np.sum(tmp_gene_df.score)
        return avgs
    def get_sigscore(self,list_index):
        list_index = sorted(list_index)
        tmp_gene_df = self.df_rank[self.df_rank.index.isin(list_index)]
        sigs = np.mean(tmp_gene_df.score) * tmp_gene_df.pos_neg.to_list().count("pos")
        return sigs

    def random_sample_rra(self,df_rank):  ###random  tag of sgrna
        null_d = []
        progress = 0
        for i in range(self.random_num):
            if progress % 10 == 0:
                print("complete permutation round:",progress)
            df_rank["Gene"] = np.random.permutation(df_rank["Gene"].values)

            null_d.append(self.get_rho_list(df_rank))
            progress += 1
        return null_d

    def random_sample_rot(self,df_rank):  ###random  tag of sgrna
        null_d = []
        progress = 0
        for i in range(self.random_num):
            if progress % 10 == 0:
                print("complete permutation round:",progress)
            df_rank["Gene"] = np.random.permutation(df_rank["Gene"].values)
    
            null_d.append(self.get_r_list(df_rank))
            progress += 1
        return null_d
    def random_sample_avgscore(self,df_rank):  ###random  tag of sgrna
        null_d = []
        progress = 0
        for i in range(self.random_num):
            if progress % 10 == 0:
                print("complete permutation round:",progress)
            df_rank["Gene"] = np.random.permutation(df_rank["Gene"].values)

            null_d.append(self.get_avgscore_list(df_rank))
            progress += 1
        return null_d
    def get_general_sg(self): ##general  sgrna number

        self.J = int(self.df_rank[self.df_rank.pos_neg=="pos"].groupby(["Gene"]).count().pos_neg.median())
        if self.J < 2:
            self.J = 2



    def run_icrisee(self):
        rra = []
        goodsg = [] 
        for gene in self.genes:
            index_list = self.df_rank[self.df_rank.Gene==gene].index
            rra.append(self.get_r(index_list))
            goodsg.append(list(self.df_rank.iloc[index_list,:].pos_neg).count("pos"))
        null_perm = np.random.choice(len(self.genes),size=(self.random_num,self.guide),replace=True)
        null_list = Parallel(n_jobs=self.num_cores)(delayed(self.get_r)(i) for i in tqdm(null_perm))
        pvalue = []
        ecdf = ECDF(null_list)
        for i in range(len(self.genes)):
            pvalue.append(ecdf(rra[i]))
        gene_list = list(self.df_rank.Gene.unique())
        gene_final = pd.DataFrame({"Gene":gene_list})
        gene_final["score"] = -np.log10(np.array(rra))
        gene_final["pvalue"] = np.array(pvalue)
#        gene_final["sgRNAS"] = self.guide
        gene_final["goodsg"] = goodsg
        gene_final = gene_final.sort_values(["pvalue","score"]).reset_index(drop=True)
        gene_final["fdr"] = gene_final.pvalue*len(self.genes)/(gene_final.index+1)
        gene_final = gene_final[~gene_final.Gene.isin(self.controlgene)]
        gene_final.to_csv(self.rotfile,sep = "\t", index=None) 

    def run_rra(self):
        rra = []
        goodsg = []
        for gene in self.genes:
            index_list = self.df_rank[self.df_rank.Gene==gene].index
            goodsg.append(self.df_rank.iloc[index_list,:].pos_neg.to_list().count("pos"))
            rra.append(self.get_rho(index_list))
        null_perm = np.random.choice(len(self.genes),size=(self.random_num,self.guide),replace=True)
        null_list = Parallel(n_jobs=self.num_cores)(delayed(self.get_rho)(i) for i in tqdm(null_perm)) 
        pvalue = []
        ecdf = ECDF(null_list)
        for i in range(len(self.genes)):
            pvalue.append(ecdf(rra[i]))
        gene_final = pd.DataFrame({"Gene":self.genes})
        gene_final["score"] = -np.log10(np.array(rra))
        gene_final["pvalue"] = np.array(pvalue)
#        gene_final["sgRNAS"] = self.guide
        gene_final["goodsg"] = goodsg
        gene_final = gene_final.sort_values(["pvalue",'score']).reset_index(drop=True)
        gene_final["fdr"] = gene_final.pvalue*len(self.genes)/(gene_final.index+1)
        gene_final = gene_final[~gene_final.Gene.isin(self.controlgene)]
        gene_final.to_csv(self.rrafile,sep = "\t", index=None)
    def run_sigscore(self):

        gene_final = pd.DataFrame({"Gene":self.genes})
        avgs = []
        goodsg = []
        pvalue=[]
        for gene in self.genes:
            index_list = self.df_rank[self.df_rank.Gene==gene].index 
            goodsg.append(list(self.df_rank.iloc[index_list,:].pos_neg).count("pos"))
            avgs.append(self.get_sigscore(index_list))
        null_perm = np.random.choice(len(self.genes),size=(self.random_num,self.guide),replace=True)
        null_list = Parallel(n_jobs=self.num_cores)(delayed(self.get_sigscore)(i) for i in tqdm(null_perm))
        ecdf = ECDF(null_list)
        for gene in range(len(self.genes)):
            pvalue.append(1-ecdf(avgs[gene]))
        gene_final["score"] = avgs
        gene_final["pvalue"] = np.array(pvalue)
#        gene_final["sgRNAS"] = self.guide
        gene_final["goodsg"] = goodsg
        gene_final = gene_final.sort_values("pvalue").reset_index(drop=True)
        gene_final["fdr"] = gene_final.pvalue*len(self.genes)/(gene_final.index+1)
        gene_final = gene_final[~gene_final.Gene.isin(self.controlgene)]
        gene_final.to_csv(self.avgsfile,sep = "\t", index=None)
    def run_sumscore(self):

       
        gene_final = pd.DataFrame({"Gene":self.genes})
        avgs = []
        pvalue=[]
        goodsg = []
        for gene in self.genes:
            index_list = self.df_rank[self.df_rank.Gene==gene].index
            goodsg.append(list(self.df_rank.iloc[index_list,:].pos_neg).count("pos"))
            avgs.append(self.get_sumscore(index_list))
        null_perm = np.random.choice(len(self.genes),size=(self.random_num,self.guide),replace=True)
        null_list = Parallel(n_jobs=self.num_cores)(delayed(self.get_sumscore)(i) for i in tqdm(null_perm))
        ecdf = ECDF(null_list)
        for gene in range(len(self.genes)):
            pvalue.append(1 - ecdf(avgs[gene]))
        gene_final["score"] = avgs
        gene_final["pvalue"] = np.array(pvalue)
#        gene_final["sgRNAS"] = self.guide
        gene_final["goodsg"] = goodsg
        gene_final = gene_final.sort_values("pvalue").reset_index(drop=True)
        gene_final["fdr"] = gene_final.pvalue*len(self.genes)/(gene_final.index+1)
        gene_final = gene_final[~gene_final.Gene.isin(self.controlgene)]
        gene_final.to_csv(self.sigsfile,sep = "\t", index=None)

if __name__ == "__main__":

    Generank("parp7","nb","grra",5,0,200)

#    Generank("cellr","nb","grra",3,0,200)
#    Generank("mela","nb","grra",5,0,20000)
#    Generank("leuke","nb","grra",10,0,200)
#    Generank("leuke3_rep4","nb","grra",3,0,200)
#    Generank("leuke4_rep4","nb","grra",4,0,200)
#    Generank("leuke5_rep4","nb","grra",5,0,200)
#    Generank("leuke6_rep4","nb","grra",6,0,200)
#    Generank("leuke7_rep4","nb","grra",7,0,200)
#    Generank("leuke8_rep4","nb","grra",8,0,200)
#    Generank("leuke9_rep4","nb","grra",9,0,200)

#    Generank("cd47","nb","grra",5,0,20000)
#    Generank("mhc1c","nb","all",5,0,200000)
#    Generank("mhc1c_neg","nb","grra",5,0,20000)
#    Generank("mhc1c_4","nb","grra",5,0,20000)
#    Generank("mhc1c_8","nb","grra",5,0,20000)
#    Generank("mhc1c_16","nb","grra",5,0,20000)
#    Generank("mhc1c_32","nb","grra",5,0,20000)





