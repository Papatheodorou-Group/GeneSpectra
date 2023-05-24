import anndata as ad
import matplotlib.pyplot as plt
import metacells as mc
import numpy as np
from numba import njit
import os
import pandas as pd
import scipy.sparse as sp
import seaborn as sb

sb.set_style("white")

import scanpy as sc

from math import hypot
from matplotlib.collections import LineCollection

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def exclude_mt_genes(input_ad, col_use=None):
    """
    Exclude genes from mitochondria

    :param input_ad: an anndata object, default input_ad.var_names is gene name
    :param col_use: alternative column in input_ad.var to get gene name (starts with MT-)
    :return: an anndata object in which genes starts with MT- are removed
    """
    if col_use is not None and not (col_use in input_ad.var.columns):
        raise KeyError(
            f"{col_use} is not in input_ad.var.columns"
        )
    if col_use is None:
        return input_ad[:, ~input_ad.var_names.str.startswith(tuple(["MT-", "Mt-", "mt-"]))]
    else:
        return input_ad[:, ~input_ad.var[col_use].str.startswith(tuple(["MT-", "Mt-", "mt-"]))]


def exclude_gene_all_zeros(input_ad):
    """
    Exclude genes with all zeros across all cells
    :param input_ad: an anndata object
    :return: an anndata object in which genes with zero counts across all cell types are removed
    """

    res_ad = sc.pp.calculate_qc_metrics(input_ad, log1p=False, inplace=False)
    res_ad = sc.pp.filter_genes(input_ad, min_counts=1, copy=True)
    return res_ad


@njit
def find_cell_cycle_gene_modules(input_ad, genes_mitotic, min_similarity_of_module=0.95):
    """

    :param min_similarity_of_module:
    :param input_ad:
    :param genes_mitotic:
    :return:
    """
    suspect_genes_mask = mc.tl.find_named_genes(input_ad, names=genes_mitotic)
    suspect_gene_names = sorted(input_ad.var_names[suspect_genes_mask])

    print("Find related gene modules")
    mc.pl.relate_genes(input_ad, random_seed=123456)

    module_of_genes = input_ad.var['related_genes_module']
    suspect_gene_modules = np.unique(module_of_genes[suspect_genes_mask])
    suspect_gene_modules = suspect_gene_modules[suspect_gene_modules >= 0]

    print(f"Suspect gene modules related to cell cycle are {suspect_gene_modules}")
    print(f"filter suspect gene modules by minimum average gene similarity={min_similarity_of_module}")

    similarity_of_genes = mc.ut.get_vv_frame(input_ad, 'related_genes_similarity')
    gene_module_use = []
    for gene_module in suspect_gene_modules:
        module_genes_mask = module_of_genes == gene_module
        similarity_of_module = similarity_of_genes.loc[module_genes_mask, module_genes_mask]
        if similarity_of_module.mean(axis=None).mean() > min_similarity_of_module:
            gene_module_use.append(gene_module)

    print(f"Forbidden gene modules related to cell cycle are {gene_module_use}")

    forbidden_genes_mask = suspect_genes_mask
    for gene_module in gene_module_use:
        module_genes_mask = module_of_genes == gene_module
        forbidden_genes_mask |= module_genes_mask
        forbidden_gene_names = sorted(input_ad.var_names[forbidden_genes_mask])
    print('Number of forbidden genes')
    print(len(forbidden_gene_names))

    return input_ad, forbidden_gene_names


@njit
def make_metacells(input_ad, forbidden_gene_name, target_umi=160000):
    """
    Using metacell2 to create metacells from raw counts of scRNA-seq data
    :param target_umi:
    :param forbidden_gene_name:
    :param input_ad: output anndata by find_cell_cycle_gene_modules, no genes with all-zero expression, no MT genes
    :return: an anndata object with each cell as a metacell
    """
    print("make_metacells")
    max_parallel_piles = mc.pl.guess_max_parallel_piles(input_ad)
    print(f"max parallel piles for metacell calculation = {max_parallel_piles}")
    mc.pl.set_max_parallel_piles(max_parallel_piles)
    target_pile_size = mc.pl.compute_target_pile_size(input_ad, target_metacell_size=target_umi)
    print(f"target pile size = {target_pile_size}")
    mc.pl.divide_and_conquer_pipeline(input_ad,
                                      min_target_pile_size=5000,
                                      max_target_pile_size=12000,
                                      forbidden_gene_names=forbidden_gene_name,
                                      random_seed=123456)

    return input_ad


def collect_metacell_annotations(input_ad, anno_col):
    """

    :param input_ad: anndata object with metacells calculated by make_metacells
    :param anno_col: column in input_ad.obs with cell type annotation
    :return: a dataframe mappinf metacells to (main) cell type in annotation
    """
    return input_ad.obs[['metacell', anno_col]].value_counts().sort_index().to_frame().reset_index()


def collect_metacells(input_ad, metacells_name):
    """

    :param input_ad: anndata object with metacells calculated by make_metacells
    :return: metacells anndata object with each cell as a metacell and outlier cells removed
    """

    return mc.pl.collect_metacells(input_ad, name=metacells_name)


def annotate_metacells_by_max(input_ad, annotation_sc, anno_col):
    """

    :param anno_col: annotation column in annotation_sc
    :param input_ad: a metacell anndata from collect_metacells
    :param annotation_sc: dataframe with cell type annotation from collect_metacell_annotations
    :return: metacells anndata object with transferred cell type annotation
    """
    annotation_sc['sc_count'] = annotation_sc[0]
    annotation = annotation_sc.query("metacell >= 0") \
        .groupby('metacell').apply(lambda x: x.loc[x['sc_count'].idxmax()]) \
        .reset_index(drop=True)[['metacell', anno_col, 'sc_count']]

    input_ad.obs['metacell'] = input_ad.obs.index
    input_ad.obs.metacell = input_ad.obs.metacell.astype("int64")
    anno_new = pd.merge(annotation, input_ad.obs)
    input_ad.obs = anno_new
    return input_ad


def check_anndata():
    print("check_anndata")


def summarize_counts():
    print("summarize_counts")


def plot_metacell_umap(input_ad, anno_col, min_dist=0.5):
    mc.pl.compute_umap_by_features(input_ad, max_top_feature_genes=1000,
                                   min_dist=min_dist, random_seed=123456)
    umap_x = mc.ut.get_o_numpy(input_ad, 'umap_x')
    umap_y = mc.ut.get_o_numpy(input_ad, 'umap_y')

    if anno_col not in input_ad.var.columns:
        raise KeyError(
            f"{anno_col} is not in input_ad.var.columns"
        )

    if anno_col is not None:
        plot = sb.scatterplot(data=input_ad.obs, x=umap_x, y=umap_y, hue=anno_col)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    else:
        plot = sb.scatterplot(x=umap_x, y=umap_y)

    return input_ad, plot


def plot_counts_density(input_ad):
    """

    :param input_ad: a raw count anndata object
    :return: a seaborn density plot
    """
    print("plot_counts_density")
    total_umis_of_cells = mc.ut.get_o_numpy(input_ad, name='__x__', sum=True)
    plot = sb.distplot(total_umis_of_cells)
    plot.set(xlabel='UMIs', ylabel='Density', yticks=[])
    return plot


@njit
def plot_gene_similarity_module(input_ad, module_num):
    """

    :param input_ad: the anndata output by find_cell_cycle_gene_modules
    :param module_num: which module to plot
    :return: a matplotlib.pyplot
    """
    if 'related_genes_module' not in input_ad.var.columns:
        raise KeyError(
            "related_genes_module is not in input_ad.var.columns, is gene modules computed by "
            "find_cell_cycle_gene_modules?"
        )

    similarity_of_genes = mc.ut.get_vv_frame(input_ad, 'related_genes_similarity')
    module_of_genes = input_ad.var['related_genes_module']

    module_genes_mask = module_of_genes == module_num
    similarity_of_module = similarity_of_genes.loc[module_genes_mask, module_genes_mask]

    similarity_of_module.index = \
        similarity_of_module.columns = [name for name in similarity_of_module.index]
    ax = plt.axes()
    sb.heatmap(similarity_of_module, vmin=0, vmax=1, xticklabels=True, yticklabels=True, ax=ax, cmap="YlGnBu")
    ax.set_title(f'Gene Module {module_num}')
    return plt
