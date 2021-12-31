import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import beta
from statsmodels.distributions.empirical_distribution import ECDF
import random
import multiprocessing
from multiprocessing import Pool
from statsmodels.sandbox.stats.multicomp import multipletests
#from scipy.interpolate import spline
import matplotlib as mpl
import matplotlib.pyplot as plt
import gseapy as gp
from gseapy.plot import barplot, dotplot
from scipy.interpolate import interpolate
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn2, venn2_circles
sns.set_style("white")
mpl.rcParams['figure.dpi'] = 600




class Plot_map:
    def __init__(self,prefix,samplelist):
        self.prefix = prefix
        self.samplelist = samplelist
        self.df = pd.read_csv("{}/map.txt".format(self.prefix),sep="\t",index_col=0)
        self.dfp = None
        self.dpi = 300
        self.plot_bar()
        self.plot_bar_percentage()
        self.plot_pie()
    def plot_bar(self):
        plt.clf()
        fig, ax = plt.subplots(figsize=[8,8])
        ax = self.df.T.plot.bar(stacked=True)
        ax.legend(bbox_to_anchor=(1.0, 1.0))
        plt.ylabel("Read Counts")
#        plt.title("Alignment Bar Plot")
        plt.savefig("{}/02map/mapped.png".format(self.prefix),bbox_inches = 'tight',dpi = self.dpi)


    def plot_bar_percentage(self):
        plt.clf()
        fig, ax = plt.subplots(figsize=[8,8])
        self.dfp =  self.df/self.df.sum()
        ax = self.dfp.T.plot.bar(stacked=True)
        ax.legend(bbox_to_anchor=(1.0, 1.0))
        plt.ylabel("Read Counts Percentage")
#        plt.title("Alignment Bar Plot(percentage)")

        plt.savefig("{}/02map/mapped_percent.png".format(self.prefix),bbox_inches = 'tight', dpi = self.dpi)
    def plot_pie(self):
        plt.clf()
        for i in range(len(self.samplelist)):
            plt.subplot(2,2,i+1) 
            self.df.iloc[:,i].plot.pie(autopct='%1.1f%%', shadow=True,label = self.samplelist[i])
#        plt.title("Alignment Pie Plot")
        plt.savefig("{}/02map/mapped_pie.png".format(self.prefix),bbox_inches = 'tight', dpi = self.dpi)

class Plot_count:
    def __init__(self,prefix,name):
        self.prefix = prefix
        self.name = name
        self.df = pd.read_csv("{}/{}".format(self.prefix,self.name),sep="\t")
        self.df_clean = pd.read_csv("{}/02sgRNA/{}_clean.tsv".format(self.prefix,self.prefix),sep="\t") 
        self.samplelist = self.df.columns[2:]
        self.dpi = 300
        self.plot_c()

    def lorenz_curve(self,X,sample):
        X_lorenz = X.cumsum()/X.sum()
        X_lorenz = np.insert(X_lorenz, 0, 0)
        X_lorenz[0], X_lorenz[-1]
        plt.clf()
        fig, ax = plt.subplots(figsize=[6,6])
    ## scatter plot of Lorenz curve
        ax.scatter(np.arange(X_lorenz.size)/(X_lorenz.size-1), X_lorenz,marker='x', color='darkgreen', s=1)
    ## line plot of equality
        ax.plot([0,1], [0,1], color='k')
        plt.xlabel('Cumulative Fraction of sgRNA', fontweight='bold')
        plt.ylabel("Cumulative Fraction of Reads", fontweight='bold')
        plt.title("Lorenz plot of sample {}\nGini:{}".format(sample,self.gini(X,sample)))
        plt.savefig("{}/01Count/lorenz_{}.png".format(self.prefix,sample),bbox_inches = 'tight', dpi = self.dpi)
    def gini(self,arr,sample):
        n = len(arr)
        coef_ = 2. / n
        const_ = (n + 1.) / n
        weighted_sum = sum([(i+1)*yi for i, yi in enumerate(arr)])
        return coef_*weighted_sum/(arr.sum()) - const_
    def dis(self,arr,sample,xlabel):
        plt.clf()
        ax = sns.distplot(arr, hist=True, kde=True,kde_kws={'linewidth': 4})
        plt.title("Reads distribution of sample {}".format(sample))
        plt.xlabel(xlabel)
        plt.ylabel("Number of sgRNAs")
        plt.savefig("{}/01Count/Distribution_{}.png".format(self.prefix,sample),bbox_inches = 'tight', dpi = self.dpi)
    def correlation(self):
        plt.clf()
        corr = self.df.corr()
        sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values,cmap="bwr")
        plt.title("Sample Correlation")
        plt.savefig("{}/01Count/samples_correlation.png".format(self.prefix),bbox_inches = 'tight', dpi = self.dpi)
        
    def plot_c(self):
############lorenz,distrubution
        for sample in self.df.columns[2:]:
            arr = np.array(sorted(self.df[sample]))
            self.lorenz_curve(arr,sample+"_raw")
            self.dis(arr,sample+"_raw","Counts per sgRNA")
            arr = np.log2(arr+1)
            self.dis(arr,sample+"_raw_log2","Log2 Counts per sgRNA")
        for sample in self.df_clean.columns[2:]:
            arr = np.array(sorted(self.df_clean[sample]))
            self.lorenz_curve(arr,sample)
            self.dis(arr,sample,"Counts per sgRNA")
            arr = np.log2(arr+1)
            self.dis(arr,sample+"_log2","Log2 Counts per sgRNA")
############correlation
        self.correlation()
class Plot_sgRNA:
    def __init__(self,prefix,name,model,controlsg=0):
        self.prefix = prefix
        self.name = name
        self.model = model
        self.controlsg = controlsg
        self.df = pd.read_csv("{}/{}".format(self.prefix,self.name),sep="\t")
        self.df_clean = pd.read_csv("{}/02sgRNA/{}_clean.tsv".format(self.prefix,self.prefix),sep="\t")
        self.sg = pd.read_csv("{}/02sgRNA/df_{}.csv".format(self.prefix,self.model),sep="\t")
        self.sg_top = self.sg.iloc[:200,:]
        self.samplelist = self.df.columns[2:]
        self.dpi = 300

        self.top_sg()
    def top_sg(self):
        plt.clf()
        plt.xlabel("log2 Counts(Control)")
        plt.ylabel("log2 Counts(Treat)")
        plt.title("sgRNA Enrichment")
#        plt.scatter(np.log2(self.sg["ctr_mean"]),np.log2(self.sg["treat_mean"]),s=2)
        
        if self.controlsg == 1:
            controlsg_list = [i.strip() for i in open("{}/controlsg.txt".format(self.prefix)).readlines()]
            controlsg_list = pd.read_csv(self.prefix+"/controlsg.txt",sep="\t",header=None).iloc[:,0].to_list()
        else:
            controlsg_list = []
#            plt.scatter(np.log2(control["ctr_mean"]),np.log2(control["treat_mean"]),s=2)
        self.sg["group"] = "Normal"
#controlsg_list = [i.strip() for i in open("controlsg.txt").readlines()]
        self.sg.loc[:200,"group"]="Top200" 
        self.sg.loc[self.sg.sgRNA.isin(controlsg_list),"group"]="Notarget"

        fig, ax = plt.subplots()
        self.sg["log2_ctr_mean"] = np.log2(self.sg["ctr_mean"])
        self.sg["log2_treat_mean"] = np.log2(self.sg["treat_mean"])
        colors = {'Normal':'grey', 'Notarget':'blue', 'Top200':'red'}
        grouped = self.sg.groupby('group')
         

        for key, group in grouped:
            print(key)
            group.plot(ax=ax, kind='scatter', x='log2_ctr_mean', y='log2_treat_mean',label=key, color=colors[key],s=2)

#        plt.scatter(np.log2(self.sg_top["ctr_mean"]),np.log2(self.sg_top["treat_mean"]),s=2,color="red")
        plt.savefig("{}/02sgRNA/scatter.png".format(self.prefix),bbox_inches = 'tight', dpi = self.dpi)
            
class Plot_gene:
    def __init__(self,prefix,name,sgrank,generank,pathway):
        self.prefix = prefix
        self.name = name
        self.sgrank = sgrank
        self.generank = generank
        self.pathway = pathway
        self.df = pd.read_csv("{}/03Gene/{}_{}.csv".format(self.prefix,self.generank,self.sgrank),sep="\t")
        self.sg = pd.read_csv("{}/02sgRNA/df_{}.csv".format(self.prefix,self.sgrank),sep="\t")
        self.gene_top = self.df.iloc[:self.pathway,:]
        self.gene_top["Gene"].to_csv("{}/03Gene/topgene.txt".format(self.prefix),header=False,index=None)
        self.dpi = 300
        
        self.top_gene()
    def top_gene(self):
        
        self.genes = list(self.df.Gene[:self.pathway])
#########top 200 gene

        for gene in self.genes:
            x = [i+1 for i in range(np.shape(self.sg)[0])]
            y = [0 for i in range(np.shape(self.sg)[0])]
            ind = []
            for index in self.sg[self.sg.Gene == gene].index:
                y[index] = 1
                ind.append(int(index))
            ind = sorted(ind)
       
            plt.clf()
            plt.plot(x,y)
            plt.xlabel("sgRNA index,sgRNA rank:{}".format("|".join([str(i) for i in ind])))
            plt.ylabel(gene)
            plt.savefig("{}/03Gene/Top/{:03d}_{}.jpg".format(self.prefix,self.genes.index(gene)+1,gene),bbox_inches = 'tight', dpi = self.dpi )


##########gene score plot
        plt.clf()
        np.random.seed(1)
        pindex = self.df.index.to_list()
        np.random.shuffle(pindex) 
        self.df["pindex"] = pindex
        self.df["group"] = "Normal"
        self.df.loc[:100,"group"] = "Topgene"
        self.df.loc[:10,"group"] = "Candidate"
        colors = {'Normal':'grey', 'Topgene':'blue',"Candidate":"red"}
        grouped = self.df.groupby('group')
        plt.title("Gene Ranking top")
        fig, ax = plt.subplots()
        for key, group in grouped:
            group.plot(ax=ax, kind='scatter', x='pindex', y='score',label=key, color=colors[key],s=2)
        for i in range(10):
            ax.annotate(self.df.loc[i,"Gene"], (self.df.loc[i,"pindex"],self.df.loc[i,"score"] ),fontsize=8,color='red')
        plt.xlabel("Gene index")
        plt.ylabel("Gene Score")
        plt.title("Top ranked gene")

#        plt.scatter(self.df.pindex,self.df["score"],s=2)
#        plt.scatter(self.gene_top.pindex,self.gene_top["score"],s=2,color="red")
        plt.savefig("{}/03Gene/Gene.jpg".format(self.prefix),bbox_inches = 'tight', dpi = self.dpi)

class Plot_pathway:
    def __init__(self,prefix,pathn):
        self.prefix = prefix
        self.pathn = pathn
        self.dpi = 300

        self.df = None
        self.run_gsea()
    def run_gsea(self):
        try:
            enr = gp.enrichr(gene_list="{}/03Gene/topgene.txt".format(self.prefix),
                     gene_sets= "KEGG_2016",#"c2.cp.kegg.v7.4.symbols.gmt",#'KEGG_2016',
                     organism='Human', # don't forget to set organism to the one you desired! e.g. Yeast
                     description="pathway",#self.prefix,
                     outdir='{}/04Pathway/'.format(self.prefix),
                     # no_plot=True,
                     cutoff=0.5 # test dataset, use lower value from range(0,1)
                    )
            colnames = ["Gene_set","Term","Overlap","P-value","aP-value","oP-value","oaP-value","Odds","Combied-score","genes"]
            self.df = pd.read_csv("{}/04Pathway/KEGG_2016.pathway.enrichr.reports.txt".format(self.prefix),sep="\t")
            self.df.columns = colnames
            self.df = self.df[["Term","aP-value","Odds","Combied-score"]]
            self.df = self.df.iloc[:10,:]
            plt.clf()
            plt.barh(self.df["Term"],-np.log10(self.df["aP-value"]))
            plt.xlabel("-log10(Adjusted P-value)")
            plt.title("Bar plot of top10 enriched pathway")
            plt.savefig("{}/04Pathway/kegg_bar.png".format(self.prefix),bbox_inches = 'tight', dpi = 200)
            plt.clf()
            g=sns.scatterplot(x=-np.log10(self.df["aP-value"]),y=self.df["Term"],hue=self.df["Combied-score"],s=self.df["Combied-score"],cmap="coolwarm")
            g.get_legend().remove()
#        plt.scatter(-np.log10(self.df["aP-value"]),self.df["Term"],s=self.df["Combied-score"]) 
            plt.xlabel("-log10(Adjusted P-value)")
            plt.title("Dot plot of top10 enriched pathway")
            plt.savefig("{}/04Pathway/kegg_dot.png".format(self.prefix),bbox_inches = 'tight', dpi = self.dpi)
        except:
            print("gsea error")

if __name__ == "__main__":
#    ma = Plot_map("MZcTEQP9TD",["Control1","Control2","Treat1","Treat2"])
#    Plot_count("leuke","leuke.csv")
#    Plot_sgRNA("leuke","leuke.csv","nb")
#    Plot_gene("leuke","leuke.csv","nb","grra",200)
#    Plot_pathway("leuke",200)
#    Plot_count("cd47","cd47.txt")
#    Plot_sgRNA("cd47","cd47.txt","nb")
    Plot_gene("cd47","cd47.txt","nb","grra",200)
#    Plot_pathway("cd47",200)
    #Plot_sgRNA("mhc1c","mhc1c.txt","nb")
#
#    Plot_gene("mhc1c",'mhc1c.txt','nb','grra',442)
#    Plot_sgRNA("mela","mela.txt","nb")
#    Plot_gene("mela",'mela.txt','nb','grra',200)
#    Plot_sgRNA("leuke","leuke.txt","nb")
#    Plot_gene("leuke",'leuke.txt','nb','grra',200)
#    Plot_gene("mhc1c",'mhc1c.txt','nb','grra',200)

#    Plot_gene("leuke",'leuke.txt','nb','arra',200)
