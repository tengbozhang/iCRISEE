import sys
import os
import sys
import scipy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.sandbox.stats.multicomp import multipletests
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from pathlib import Path
class Build_model:
    def __init__(self,count,prefix="test",ctr=[2,3],treat=[4,5],basicnum = 0,controlsg=1,alpha = 0.05,padj= "fdr_tsbh",delta = 1):
        self.alpha = alpha
        self.prefix = prefix
      
        self.padj = padj
        self.delta = delta 
        self.count = count
        self.ctr = ctr
        self.treat = treat
        
        Path(prefix).mkdir(parents=True, exist_ok=True)
        Path(prefix+"/01Count").mkdir(parents=True, exist_ok=True)
        Path(prefix+"/02sgRNA").mkdir(parents=True, exist_ok=True)
        Path(prefix+"/03Gene/Top").mkdir(parents=True, exist_ok=True)
        Path(prefix+"/04Pathway").mkdir(parents=True, exist_ok=True)
        self.df = None
        self.basicnum = basicnum
        self.ctr_mean = None
        self.ctr_var = None
        self.model_ctr_mean = None
        self.model_ctr_var = None
        self.treat_mean = None
        if controlsg == 1:
            self.controlsg = pd.read_csv(self.prefix+"/controlsg.txt",sep="\t",header=None).iloc[:,0].to_list()
        else:
            self.controlsg = pd.read_csv("{}".format(self.count),sep="\t").iloc[:,0].to_list()
        self.norfile = "{}/02sgRNA/{}_nor.tsv".format(self.prefix,self.prefix)
        self.cleanfile = "{}/02sgRNA/{}_clean.tsv".format(self.prefix,self.prefix)
        self.clean()
        self.normalize_data()
        self.get_ctr_mean_variance()
        
        self.fc_fname = "{}/02sgRNA/df_fc.csv".format(self.prefix)
        self.nb_fname = "{}/02sgRNA/df_nb.csv".format(self.prefix)
        self.pos_fname = "{}/02sgRNA/df_pos.csv".format(self.prefix)
        self.run()
    def clean(self):
        self.df = pd.read_csv(self.count, sep = "\t").sort_values(["Gene"])
#####drop gene with half value less than basicnum
        self.df["Gene"] = self.df["Gene"].astype(str)
        self.df["sgRNA"] = self.df["sgRNA"].astype(str)
        gene_index=[]
        for i in self.df.Gene.unique():
            ano = self.df[self.df.Gene == i].iloc[:,2:].values.flatten()
            if len(ano[ano < self.basicnum]) > 0.5 * len(ano):# or len(ann[ann < basicnum]) >= 0.5 * len(ann):
                gene_index.append(i)  
        self.df = self.df[~self.df.Gene.isin(gene_index)].reset_index(drop=True)
#####drop sgrna with half control value less than basicnum
        index_none = []
        for index in self.df.index:
            ano = self.df.iloc[index,self.ctr].values.flatten() 
            ann = self.df.iloc[index,self.treat].values.flatten()        
            if len(ano[ano < self.basicnum]) > 0.5 * len(ano) or len(ann[ann < self.basicnum]) > 0.5 * len(ann):
                index_none.append(index)
        self.df = self.df[~self.df.index.isin(index_none)].reset_index(drop=True)
        self.df.to_csv(self.cleanfile, sep = "\t", index = None)
        print("data cleaned")

    def normalize_data(self):
        Counts_array = self.df.iloc[0:,2:]
        control_sg = self.df[self.df.sgRNA.isin(self.controlsg)].iloc[0:,2:]
        Counts_array = Counts_array/control_sg.sum()*np.mean(np.array(control_sg.sum()))
        Counts_array = Counts_array+ 1
        self.df.iloc[:,2:] = Counts_array
        self.df = self.df.reset_index(drop=True) 
        self.df.to_csv(self.norfile,sep="\t",index=None)
        print("data normalized")
    def get_ctr_mean_variance(self):
        if len(self.ctr) == 1:
            self.ctr_mean = self.df.iloc[:,self.ctr[0]].to_list()
            self.model_ctr_mean = list(np.mean(self.df[self.df.sgRNA.isin(self.controlsg)].iloc[:,2:],axis=1)) 
            self.model_ctr_var = list(np.var(self.df[self.df.sgRNA.isin(self.controlsg)].iloc[:,2:],axis=1))
        else:
            self.ctr_mean = list(np.mean(self.df.iloc[:,self.ctr],axis=1))
            self.model_ctr_mean = list(np.mean(self.df[self.df.sgRNA.isin(self.controlsg)].iloc[:,self.ctr],axis=1))
            self.model_ctr_var = list(np.var(self.df[self.df.sgRNA.isin(self.controlsg)].iloc[:,self.ctr],axis=1))
        if len(self.treat) == 1:
            self.treat_mean = self.df.iloc[:,self.treat[0]].to_list()
        else:
            self.treat_mean = list(np.mean(self.df.iloc[:,self.treat],axis=1))
        print("mean variance")
#fucntion input:control mean variance,treat mean,ouput: n,p,pvalue
    def fc_model(self):

        df_fc = self.df.iloc[:,:2]
        df_fc["ctr_mean"] = self.ctr_mean
        df_fc["treat_mean"] = self.treat_mean
        df_fc["fc"] = np.array(self.treat_mean)/np.array(self.ctr_mean)
        df_fc["pvalue"] = 1/df_fc["fc"]
        df_fc["score"] = np.log2(np.array(self.treat_mean)/np.array(self.ctr_mean))
        df_fc = df_fc.sort_values(["pvalue"]).reset_index(drop=True)
        df_fc.to_csv(self.fc_fname,sep = "\t",index = None)
        print("fc model")
    def pos_model(self):
        pval = []
        for i in range(len(self.treat_mean)):
            pval.append(1 - scipy.stats.poisson.cdf(np.round(self.treat_mean[i]),np.round(self.ctr_mean[i])))
        
        df_po = self.df.iloc[:,:2]
        df_po["ctr_mean"] = self.ctr_mean
        df_po["treat_mean"] = self.treat_mean
        df_po["fc"] = np.mean(self.treat_mean)/np.mean(self.ctr_mean)
        df_po["rawpvalue"] = np.array(pval)+10**-100
        df_po["score"] = -np.log10(df_po["rawpvalue"])
        df_po = df_po.sort_values(["rawpvalue","fc"],ascending=[True, False]).reset_index(drop=True)
        df_po.to_csv(self.pos_fname,sep = "\t",index = None)
        print("pos model")
    def nb_model(self):
        print("len model_ctr_mean:{}".format(len(self.model_ctr_mean)))
        print("len model_ctr_var:{}".format(len(self.model_ctr_var)))
        x = [np.log(self.model_ctr_mean[i]) for i in range(len(self.model_ctr_mean)) if  self.model_ctr_var[i]>=self.model_ctr_mean[i]]
##log var - mean
        print("x:{}".format(len(x)))
        y = [np.log(self.model_ctr_var[i]-self.model_ctr_mean[i]) for i in range(len(self.model_ctr_mean)) if self.model_ctr_var[i]>=self.model_ctr_mean[i]]
        print("y:{}".format(len(y)))
        reg = LinearRegression().fit(np.array(x).reshape(-1, 1), np.array(y))
#   b,log(k)    reg.coef_,reg.intercept_
        b,m=reg.coef_,reg.intercept_
        coef = reg.score(np.array(x).reshape(-1, 1), y)
#d2 = mean+e**(log(k)+b*log(mean))
#    D = np.exp(np.sum([y[i]-2*x[i] for i in range(len(x))])/len(model_ctr_mean))
#    ctr_var = model_ctr_mean + D * model_ctr_mean**k
        self.ctr_var = self.ctr_mean+np.exp(reg.intercept_)*(self.ctr_mean**reg.coef_)
        n,p=[],[]
        for i in range(len(self.ctr_mean)):
            if self.ctr_mean[i]==0 or self.ctr_var[i]==0:
                n.append(((self.ctr_mean[i]+self.delta)**2/(self.ctr_var[i]+self.delta))/(1-(self.ctr_mean[i]+self.delta)/(self.ctr_var[i]+self.delta)))
                p.append((self.ctr_mean[i]+self.delta)/(self.ctr_var[i]+self.delta))
            else:
                n.append((self.ctr_mean[i]**2/self.ctr_var[i])/(1-self.ctr_mean[i]/self.ctr_var[i]))
                p.append(self.ctr_mean[i]/self.ctr_var[i])
        pval = []
        for i in range(len(self.treat_mean)):
            pval.append(1 - scipy.stats.nbinom.cdf(self.treat_mean[i],n[i],p[i]))

        df_nb = self.df.iloc[:,:2]
        df_nb["ctr_mean"] = self.ctr_mean
        df_nb["treat_mean"] = self.treat_mean
        df_nb["nb.n"] = n
        df_nb["nb.p"] = p
        df_nb["pvalue"] = pval #rawpvalue
        df_nb["score"] = -np.log10(df_nb["pvalue"])
        df_nb["fc"] = np.array(self.treat_mean)/np.array(self.ctr_mean)
#    df_simple_nb["pvalue"] = multipletests(pval,alpha,padj)[1]
        df_nb = df_nb.sort_values(["pvalue","fc"],ascending=[True, False]).reset_index(drop=True)
        df_nb.to_csv(self.nb_fname,sep = "\t",index = None)
        print("nb model")
    def run(self):
        self.fc_model()
        self.nb_model()
        self.pos_model()



####main workflow
if __name__ == "__main__":
    #Build_model("mhc1c/mhc1c.txt","mhc1c",ctr=[2],treat=[3],basicnum = 0,controlsg=0)
#    Build_model("mhc1c/mhc1c.txt","mhc1c_neg",ctr=[2],treat=[3],basicnum = -10,controlsg=0)
    Build_model("leuke3_rep4/leuke.txt","leuke3_rep4",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
    Build_model("leuke4_rep4/leuke.txt","leuke4_rep4",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
    Build_model("leuke5_rep4/leuke.txt","leuke5_rep4",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
    Build_model("leuke6_rep4/leuke.txt","leuke6_rep4",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
    Build_model("leuke7_rep4/leuke.txt","leuke7_rep4",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
    Build_model("leuke8_rep4/leuke.txt","leuke8_rep4",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
    Build_model("leuke9_rep4/leuke.txt","leuke9_rep4",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
#    Build_model("leuke/leuke.txt","leuke",ctr=[2,3],treat=[4,5],basicnum = 0,controlsg=0)

#    Build_model("mhc1c/mhc1c.txt","mhc1c_4",ctr=[2],treat=[3],basicnum = 4,controlsg=0)
#    Build_model("mhc1c/mhc1c.txt","mhc1c_8",ctr=[2],treat=[3],basicnum = 8,controlsg=0)
#    Build_model("mhc1c/mhc1c.txt","mhc1c_16",ctr=[2],treat=[3],basicnum = 16,controlsg=0)
#    Build_model("mhc1c/mhc1c.txt","mhc1c_32",ctr=[2],treat=[3],basicnum = 32,controlsg=0)

#    Build_model("cd47/cd47.csv","cd47",ctr=[2],treat=[3],basicnum = 1,controlsg=1)

#    Build_model("mela/mela.txt","mela",ctr=[9,10],treat=[2],basicnum = 1,controlsg=0)
#    Build_model("dD719fJ24tPInKKrW4vR/dD719fJ24tPInKKrW4vR.csv","dD719fJ24tPInKKrW4vR",ctr=[2,3],treat=[4,5],basicnum = 1,controlsg="controlsg.txt")
#    Build_model("leuke3/leuke3.txt","leuke3",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
#    Build_model("leuke4/leuke4.txt","leuke4",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
#    Build_model("leuke5/leuke5.txt","leuke5",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
#    Build_model("leuke6/leuke6.txt","leuke6",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
#    Build_model("leuke7/leuke7.txt","leuke7",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
#    Build_model("leuke8/leuke8.txt","leuke8",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
#    Build_model("leuke9/leuke9.txt","leuke9",ctr=[4,5],treat=[2,3],basicnum = 0,controlsg=0)
#run_model_all(filename = "analysis/leu_clean_nor.csv")   






