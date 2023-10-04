# %%
import numpy as np
import pandas as pd
from fast_matrix_market import mmread
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix

# %%
# read indices and columns
path = "data/celegans/annotations.csv"
df = pd.read_csv(path, sep=",", dtype="str")["cell"]
cells = df.values

path = "data/celegans/genes.csv"
df = pd.read_csv(path, sep=",", dtype="str")["gene_short_name"]
genes = df.values

print(len(cells), "cells:", cells[:4], "\n")
print(len(genes), "genes:", genes[:5])

# %%
# read counts
data = mmread("data/celegans/counts.mtx").transpose()
data = csr_matrix(data, dtype=np.float32)
data = ad.AnnData(X=data)
data.obs_names = cells
data.var_names = genes
print("Original data shape:", data.shape)

# %%
# Columns of interest
ct = "plot.cell.type"
c1 = "n.umi"
c2 = "embryo.time.bin"
c3 = "lineage"
c4 = "batch"
c5 = "passed_initial_QC_or_later_whitelisted"
cols = [ct, c1, c2, c3, c4, c5]

design_file_path = "data/celegans/annotations.csv"
df = pd.read_csv(design_file_path, sep=",", index_col="cell", dtype="str")[cols].astype("str")
df.rename(columns={ct: "cell_type", c1: "n_umi", c2: "time", c3: "lineage", c4: "batch", c5: "qc_pass"}, inplace=True)
# df.index = df.index.map(lambda x: x.split("-")[0])
print(df["cell_type"].value_counts())

# "cluster" can be either "WT microglia", "Irf8 KO microglia", "WT CP-BAM" or "Irf8 KO CP-BAM"
df["qc_pass"] = df["qc_pass"].apply(lambda x: 1 if x=="TRUE" else 0)
print("\n", df["qc_pass"].value_counts(), "\n")

data = data[df.index] # index data object with the valid cells which have annotations
data.obs = df 
data.var_names_make_unique()

# %%
sc.pp.filter_cells(data, min_genes=200)
sc.pp.filter_cells(data, min_counts=500)
sc.pp.filter_cells(data, max_genes=5000) 

# Filter out genes that are not expressed in a minimum number of cells
sc.pp.filter_genes(data, min_cells=5)

# save data 
data.write("data/celegans/anndata.h5ad")
print("Saved data:", data.shape)


# %%
# optionally take only highly variable genes
sc.pp.log1p(data)
sc.pp.highly_variable_genes(data)

# filter and undo log
data = data[:, data.var['highly_variable']]
data.X = np.expm1(data.X)

# save hvg data
data.write("data/celegans/anndata_hvg.h5ad")
print("Saved hvg data:", data.shape)


