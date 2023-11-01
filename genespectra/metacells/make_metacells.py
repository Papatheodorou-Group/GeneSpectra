import matplotlib.pyplot as plt
import metacells as mc
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from typing import Tuple, List

sns.set_style("white")
import scanpy as sc
import anndata as ad
from anndata import AnnData
from math import hypot
from matplotlib.collections import LineCollection
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

forbidden_gene_names = []


def exclude_mt_genes(
        adata: AnnData,
        col_use: str = None
) -> AnnData:
    """
    Exclude genes from mitochondria

    :param adata: an anndata object, default adata.var_names is gene name
    :param col_use: alternative column in adata.var to get gene name (starts with MT-)
    :return: an anndata object in which genes starts with MT- are removed
    """
    if col_use is not None and not (col_use in adata.var.columns):
        raise KeyError(
            f"{col_use} is not in adata.var.columns"
        )
    if col_use is None:
        return adata[:, ~adata.var_names.str.startswith(tuple(["MT-", "Mt-", "mt-"]))]
    else:
        return adata[:, ~adata.var[col_use].str.startswith(tuple(["MT-", "Mt-", "mt-"]))]


def exclude_gene_all_zeros(
        adata: AnnData
) -> AnnData:
    """
    Exclude genes with all zeros across all cells
    :param adata: an anndata object
    :return: an anndata object in which genes with zero counts across all cell types are removed
    """

    res_ad = sc.pp.calculate_qc_metrics(adata, log1p=False, inplace=False)
    res_ad = sc.pp.filter_genes(adata, min_counts=1, copy=True)
    return res_ad


def find_cell_cycle_gene_modules(
        adata: AnnData,
        genes_mitotic: list,
        min_similarity_of_module: float = 0.95
) -> Tuple[AnnData, List[int]]:
    """

    :param min_similarity_of_module:
    :param adata:
    :param genes_mitotic:
    :return:
    """
    global forbidden_gene_names
    ## is this correct?

    suspect_genes_mask = mc.tl.find_named_genes(adata, names=genes_mitotic)
    suspect_gene_names = sorted(adata.var_names[suspect_genes_mask])

    print("Find related gene modules")
    mc.pl.relate_genes(adata, random_seed=123456)

    module_of_genes = adata.var['related_genes_module']
    suspect_gene_modules = np.unique(module_of_genes[suspect_genes_mask])
    suspect_gene_modules = suspect_gene_modules[suspect_gene_modules >= 0]

    print(f"Suspect gene modules related to cell cycle are {suspect_gene_modules}")
    print(f"filter suspect gene modules by minimum average gene similarity={min_similarity_of_module}")

    similarity_of_genes = mc.ut.get_vv_frame(adata, 'related_genes_similarity')
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
        forbidden_gene_names = sorted(adata.var_names[forbidden_genes_mask])
    print('Number of forbidden genes')
    print(len(forbidden_gene_names))

    return adata, forbidden_gene_names


def make_metacells(
        adata: AnnData,
        forbidden_gene_names: list,
        target_umi: int = 160000
) -> AnnData:
    """
    Using metacell2 to create metacells from raw counts of scRNA-seq data
    :param target_umi:
    :param forbidden_gene_names:
    :param adata: output anndata by find_cell_cycle_gene_modules, no genes with all-zero expression, no MT genes
    :return: an anndata object with metacell information, each cell is still a single cell at this point
    """
    print("make_metacells")
    max_parallel_piles = mc.pl.guess_max_parallel_piles(adata)
    print(f"max parallel piles for metacell calculation = {max_parallel_piles}")
    mc.pl.set_max_parallel_piles(max_parallel_piles)
    target_pile_size = mc.pl.compute_target_pile_size(adata, target_metacell_size=target_umi)
    print(f"target pile size = {target_pile_size}")
    mc.pl.divide_and_conquer_pipeline(adata,
                                      min_target_pile_size=5000,
                                      max_target_pile_size=12000,
                                      forbidden_gene_names=forbidden_gene_names,
                                      random_seed=123456)

    return adata


def collect_metacell_annotations(
        adata: AnnData,
        anno_col: str
) -> pd.DataFrame:
    """

    :param adata: anndata object with metacells calculated by make_metacells
    :param anno_col: column in adata.obs with cell type annotation
    :return: a dataframe mapping metacells to (main) cell type in annotation
    """
    return adata.obs[['metacell', anno_col]].value_counts().sort_index().to_frame().reset_index()


def collect_metacells(
        adata: AnnData,
        metacells_name: str
) -> AnnData:
    """

    :param metacells_name: give metacells dataset a name
    :param adata: anndata object with metacells calculated by make_metacells
    :return: metacells anndata object with each cell as a metacell and outlier cells removed
    """

    return mc.pl.collect_metacells(adata, name=metacells_name)


def annotate_metacells_by_max(
        adata: AnnData,
        annotation_sc: pd.DataFrame,
        anno_col: str
) -> AnnData:
    """

    :param anno_col: annotation column in annotation_sc
    :param adata: a metacell anndata from collect_metacells
    :param annotation_sc: dataframe with cell type annotation from collect_metacell_annotations
    :return: metacells anndata object with transferred cell type annotation
    """
    annotation_sc['sc_count'] = annotation_sc['count'].values
    annotation = annotation_sc.query("metacell >= 0") \
        .groupby('metacell').apply(lambda x: x.loc[x['sc_count'].idxmax()]) \
        .reset_index(drop=True)[['metacell', anno_col, 'sc_count']]

    adata.obs['metacell'] = adata.obs.index
    adata.obs.metacell = adata.obs.metacell.astype("int64")
    anno_new = pd.merge(annotation, adata.obs)
    adata.obs = anno_new

    # set index dtype to str

    adata.obs.index = adata.obs.index.astype("str")

    return adata


def check_anndata():
    print("check_anndata")


def summarize_counts():
    print("summarize_counts")


def plot_metacell_umap(
        adata: AnnData,
        anno_col: str,
        min_dist: float = 0.5
) -> Tuple[AnnData, plt.Axes]:
    """

    :param adata:
    :param anno_col:
    :param min_dist:
    :return:
    """
    mc.pl.compute_umap_by_features(adata, max_top_feature_genes=1000,
                                   min_dist=min_dist, random_seed=123456)
    umap_x = mc.ut.get_o_numpy(adata, 'umap_x')
    umap_y = mc.ut.get_o_numpy(adata, 'umap_y')

    if anno_col not in adata.obs.columns:
        raise KeyError(
            f"{anno_col} is not in adata.obs.columns"
        )

    if anno_col is not None:
        plot = sns.scatterplot(data=adata.obs, x=umap_x, y=umap_y, hue=anno_col)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    else:
        plot = sns.scatterplot(x=umap_x, y=umap_y)

    return adata, plot


def plot_counts_density(
        adata: AnnData,
) -> plt.Axes:
    """

    :param adata: a raw count anndata object
    :return: a seaborn density plot
    """
    print("plot_counts_density")
    total_umis_of_cells = mc.ut.get_o_numpy(adata, name='__x__', sum=True)
    plot = sns.distplot(total_umis_of_cells)
    plot.set(xlabel='UMIs', ylabel='Density', yticks=[])
    return plot


def plot_gene_similarity_module(
        adata: AnnData,
        module_num: int,
) -> plt.Axes:
    """

    :param adata: the anndata output by find_cell_cycle_gene_modules
    :param module_num: which module to plot
    :return: a matplotlib.pyplot
    """
    if 'related_genes_module' not in adata.var.columns:
        raise KeyError(
            "related_genes_module is not in adata.var.columns, is gene modules computed by "
            "find_cell_cycle_gene_modules?"
        )

    similarity_of_genes = mc.ut.get_vv_frame(adata, 'related_genes_similarity')
    module_of_genes = adata.var['related_genes_module']

    module_genes_mask = module_of_genes == module_num
    similarity_of_module = similarity_of_genes.loc[module_genes_mask, module_genes_mask]

    similarity_of_module.index = \
        similarity_of_module.columns = [name for name in similarity_of_module.index]
    ax = plt.axes()
    sns.heatmap(similarity_of_module, vmin=0, vmax=1, xticklabels=True, yticklabels=True, ax=ax, cmap="YlGnBu")
    ax.set_title(f'Gene Module {module_num}')
    return ax


def plot_num_sc_per_metacell(
        annotation_sc: pd.DataFrame
) -> plt.Axes:
    """

    :param annotation_sc:
    :return:
    """
    plot = plt.axes()
    plt.hist(annotation_sc.query("metacell >= 0")['count'], bins=30)
    plt.title("Number of single cells per metacell")

    return plot


def plot_num_cell_type_sc_per_metacell(
        metacells_res: AnnData,
        anno_col: str,
) -> plt.Axes:
    """

    :param metacells_res:
    :param anno_col:
    :return:
    """
    if 'metacell' not in metacells_res.obs.columns:
        raise KeyError("metacell not in metacells_res.obs.columns")

    if anno_col not in metacells_res.obs.columns:
        raise KeyError(
            f"{anno_col} is not in adata.obs.columns"
        )

    df = metacells_res.obs[['metacell', anno_col]].query("metacell >= 0") \
        .groupby("metacell")[anno_col].nunique()

    plot = plt.axes()
    plt.hist(df, bins=10)
    plt.title("Number of cell types of single cells per metacell")

    return plot


def make_metacells_one_cell_type(adata: AnnData,
                                 cell_type_now: str,
                                 annotation_col: str,
                                 forbidden_gene_names: list,
                                 target_umi: int = 160000) -> AnnData:
    adata_now = adata[adata.obs[annotation_col] == cell_type_now, :]

    max_parallel_piles = mc.pl.guess_max_parallel_piles(adata_now)
    print(f"max parallel piles for metacell calculation = {max_parallel_piles}")

    mc.pl.set_max_parallel_piles(max_parallel_piles)
    target_pile_size = mc.pl.compute_target_pile_size(adata_now, target_metacell_size=target_umi)

    print(f"target pile size = {target_pile_size}")

    # this function does not return anything
    mc.pl.divide_and_conquer_pipeline(adata_now, min_target_pile_size=5000, max_target_pile_size=12000,
                                      forbidden_gene_names=forbidden_gene_names, random_seed=123456)
    adata_now_mc = mc.pl.collect_metacells(adata_now, name=cell_type_now)
    adata_now_mc.obs['cell_type'] = cell_type_now
    adata_now_mc.obs['cell_name'] = adata_now_mc.obs.apply(lambda row: row.cell_type + "_" + str(row.candidate), axis=1)
    adata_now_mc.obs_names = adata_now_mc.obs['cell_name'].values

    return adata_now_mc


def make_metacells_per_group(adata: AnnData,
                             annotation_col: str,
                             forbidden_gene_names: list,
                             target_umi: int = 160000) -> AnnData:
    all_cell_type_mc_adatas = list()
    all_feature_genes = list()
    for cell_type_now in adata.obs[annotation_col].unique():
        print(f"making metacells for {cell_type_now}")
        cell_type_now_mc_adata = make_metacells_one_cell_type(adata, cell_type_now, annotation_col,
                                                              forbidden_gene_names, target_umi=target_umi)
        feature_genes_now = cell_type_now_mc_adata.var.loc[
            cell_type_now_mc_adata.var.feature_gene == 1].index.values
        all_feature_genes.extend(feature_genes_now)
        all_cell_type_mc_adatas.append(cell_type_now_mc_adata)

    final_mc_adata = ad.concat(all_cell_type_mc_adatas, axis=0, join='outer', merge='same')

    all_feature_genes = set(all_feature_genes)

    final_mc_adata.var['feature_gene'] = False

    final_mc_adata.var.loc[final_mc_adata.var_names.isin(all_feature_genes), 'feature_gene'] = True

    return final_mc_adata


class SummedAnnData(AnnData):
    def __init__(self, summed_adata=None, removed_genes=None, removed_min_count=None):
        super().__init__(X=summed_adata.X if summed_adata.X is not None else None,
                         obs=summed_adata.obs if summed_adata.obs is not None else None,
                         var=summed_adata.var if summed_adata.var is not None else None)

        self._count_type = "summed_counts"
        self.removed_genes = removed_genes
        self.removed_min_count = removed_min_count

    @property
    def count_type(self):
        return self._count_type

    @classmethod
    def create_from_anndata(cls, adata, annotation_col, removed_genes=None, removed_min_count=None):
        summed_adata = sum_expression_by_class(adata=adata, annotation_col=annotation_col)
        return cls(summed_adata, removed_genes, removed_min_count)

    def filter_low_counts(self, min_count):
        assert isinstance(min_count, (int, float)), 'min_count should be int or float type'
        print(f"Genes with min_count {min_count} are considered low count")
        summed_ad_filtered = sc.pp.filter_genes(self, min_counts=min_count, copy=True)
        removed_genes = [x for x in self.var_names.values if x not in summed_ad_filtered.var_names.values]
        summed_ad_filtered.removed_genes = removed_genes
        summed_ad_filtered.removed_min_count = min_count
        print(f"Put {len(removed_genes)} genes into low counts genes")
        return summed_ad_filtered

    def depth_normalize_counts(self, target_sum=None):

        print("Size factor depth normalize counts")
        if target_sum is not None:
            print(f"Target total UMI per cell is {target_sum}")
        else:
            print(f"Target total UMI per cell is the average UMI across cells")
        sc.pp.normalize_total(self, target_sum=target_sum, inplace=True)
        print(f"Total UMI count is normalized to {self.X.sum(axis=1)[0].round()}")

        return self


def sum_expression_by_class(adata, annotation_col):
    """
    Aggregate scRNA-seq data by cell class, create a "pseudo-bulk"
    :param adata: input anndata object
    :param annotation_col: the column in adata.obs indicating cell groups to combine
    :return: an anndata object storing combined counts for each cell group
    """
    # Create a new AnnData object to store the summed expression levels
    summed_adata = AnnData()

    # numerical class id does not work
    adata.obs[annotation_col] = adata.obs[annotation_col].astype("str")

    # Iterate over the unique classes in the given class key and sum the counts
    unique_classes = adata.obs[annotation_col].unique()
    for class_value in unique_classes:
        class_cells = adata[adata.obs[annotation_col] == class_value]

        summed_expression = class_cells.X.sum(axis=0)

        temp_adata = AnnData(X=summed_expression.reshape(1, -1))
        temp_adata.obs[annotation_col] = class_value
        temp_adata.obs_names = [class_value]

        if summed_adata.X is None:
            summed_adata = temp_adata
        else:
            summed_adata = ad.concat([summed_adata, temp_adata])

    summed_adata.var = adata.var

    # required for sc.pp.calculate_qc_metrics
    summed_adata.X = np.array(summed_adata.X)

    return summed_adata
