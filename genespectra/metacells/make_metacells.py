import matplotlib.pyplot as plt
import metacells as mc
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns

sns.set_style("white")
import scanpy as sc
import anndata as ad
from anndata import AnnData
from math import hypot
from matplotlib.collections import LineCollection
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

random_seed = 123456


def exclude_mt_genes(adata: AnnData, col_use: str = None) -> AnnData:
    """
    Exclude genes from mitochondria

    :param adata: an anndata object, default adata.var_names is gene name
    :param col_use: alternative column in adata.var to get gene name (starts with MT-)
    :return: an anndata object in which genes starts with MT- are removed
    """
    if col_use is not None and not (col_use in adata.var.columns):
        raise KeyError(f"{col_use} is not in adata.var.columns")
    if col_use is None:
        return adata[
            :, ~adata.var_names.str.startswith(tuple(["MT-", "Mt-", "mt-"]), na=False)
        ]
    else:
        return adata[
            :,
            ~adata.var[col_use].str.startswith(tuple(["MT-", "Mt-", "mt-"]), na=False),
        ]
    # if the column value is nan, return it (do not remove)


def exclude_gene_all_zeros(adata: AnnData) -> AnnData:
    """
    Exclude genes with all zeros across all cells
    :param adata: an anndata object
    :return: an anndata object in which genes with zero counts across all cell types are removed
    """

    sc.pp.calculate_qc_metrics(adata, log1p=False, inplace=True)
    sc.pp.filter_genes(adata, min_counts=1, inplace=True)
    return adata


def find_cell_cycle_gene_modules(
    adata: AnnData, genes_mitotic: list, min_similarity_of_module: float = 0.95
) -> tuple[AnnData, list[int]]:
    """
    Find cell cycle gene modules related to mitotic genes, requiring minimum similarity of gene modules,
    this is a shortcut and it is recommended to use the metacell workflow to decide on whether to exclude each module or not
    :param min_similarity_of_module: minimum average gene similarity of a module to be excluded
    :param adata: input anndata object with genes with all-zero expression removed, MT genes removed
    :param genes_mitotic: referece mitotic genes, typically obtained manually or via GO terms
    :return: an anndata object with cell cycle-related gene names marked, and a list of forbidden gene names
    """
    forbidden_gene_names = []
    global random_seed

    suspect_genes_mask = mc.tl.find_named_genes(adata, names=genes_mitotic)

    print("using adata var_name to match forbidden gene names")
    suspect_gene_names = sorted(adata.var_names[suspect_genes_mask])

    print("Find related gene modules")
    mc.pl.relate_genes(adata, random_seed=random_seed)

    module_of_genes = adata.var["related_genes_module"]
    suspect_gene_modules = np.unique(module_of_genes[suspect_genes_mask])
    suspect_gene_modules = suspect_gene_modules[suspect_gene_modules >= 0]

    print(f"Suspect gene modules related to cell cycle are {suspect_gene_modules}")
    print(
        f"filter suspect gene modules by minimum average gene similarity={min_similarity_of_module}"
    )

    similarity_of_genes = mc.ut.get_vv_frame(adata, "related_genes_similarity")
    gene_module_use = []
    for gene_module in suspect_gene_modules:
        module_genes_mask = module_of_genes == gene_module
        similarity_of_module = similarity_of_genes.loc[
            module_genes_mask, module_genes_mask
        ]
        if similarity_of_module.mean(axis=None).mean() > min_similarity_of_module:
            gene_module_use.append(gene_module)

    print(f"Forbidden gene modules related to cell cycle are {gene_module_use}")

    forbidden_genes_mask = suspect_genes_mask

    if len(gene_module_use) == 0:
        print("No gene module is excluded")
        # then only exclude known genes that are related to mitosis
        forbidden_gene_names = genes_mitotic

    for gene_module in gene_module_use:
        module_genes_mask = module_of_genes == gene_module
        forbidden_genes_mask |= module_genes_mask
        forbidden_gene_names = sorted(adata.var_names[forbidden_genes_mask])
    print("Number of forbidden genes")
    print(len(forbidden_gene_names))

    return adata, forbidden_gene_names


def make_metacells(
    adata: AnnData, forbidden_gene_names: list[str], target_umi: int = 160000
) -> AnnData:
    """
    Using metacell2 to create metacells from raw counts of scRNA-seq data
    :param target_umi: target UMI per metacell
    :param forbidden_gene_names: list of forbidden gene names, these genes are excluded in metacell construction
    :param adata: output anndata by find_cell_cycle_gene_modules, no genes with all-zero expression, no MT genes
    :return: an anndata object with metacell information, each cell is still a single cell at this point just assigned to a metacell
    """
    print("make_metacells")

    global random_seed
    max_parallel_piles = mc.pl.guess_max_parallel_piles(adata)
    print(f"max parallel piles for metacell calculation = {max_parallel_piles}")
    mc.pl.set_max_parallel_piles(max_parallel_piles)
    target_pile_size = mc.pl.compute_target_pile_size(
        adata, target_metacell_size=target_umi
    )
    print(f"target pile size = {target_pile_size}")
    mc.pl.divide_and_conquer_pipeline(
        adata,
        min_target_pile_size=5000,
        max_target_pile_size=12000,
        forbidden_gene_names=forbidden_gene_names,
        random_seed=random_seed,
    )

    return adata


def collect_metacell_annotations(adata: AnnData, anno_col: str) -> pd.DataFrame:
    """
    Get a summary table for how many single cells per metacell
    :param adata: anndata object with metacells calculated by make_metacells
    :param anno_col: column in adata.obs with cell type annotation
    :return: a dataframe mapping metacells to (main) cell type in annotation
    """
    return (
        adata.obs[["metacell", anno_col]]
        .value_counts()
        .sort_index()
        .to_frame()
        .reset_index()
    )


def collect_metacells(adata: AnnData, metacells_name: str) -> AnnData:
    """
    Collect metacells from anndata object
    :param metacells_name: give metacells dataset a name
    :param adata: anndata object with metacells calculated by make_metacells
    :return: metacells anndata object with each cell as a metacell and outlier cells removed
    """

    return mc.pl.collect_metacells(adata, name=metacells_name)


def annotate_metacells_by_max(
    adata: AnnData, annotation_sc: pd.DataFrame, anno_col: str
) -> AnnData:
    """
    Transfer annotation from single cells to metacells by majority vote

    :param anno_col: annotation column in annotation_sc
    :param adata: a metacell anndata from collect_metacells
    :param annotation_sc: dataframe with cell type annotation from collect_metacell_annotations
    :return: metacells anndata object with transferred cell type annotation
    """
    annotation_sc["sc_count"] = annotation_sc["count"].values
    annotation = (
        annotation_sc.query("metacell >= 0")
        .groupby("metacell")
        .apply(lambda x: x.loc[x["sc_count"].idxmax()])
        .reset_index(drop=True)[["metacell", anno_col, "sc_count"]]
    )

    adata.obs["metacell"] = adata.obs.index
    adata.obs.metacell = adata.obs.metacell.astype("int64")
    anno_new = pd.merge(annotation, adata.obs)
    adata.obs = anno_new

    # set index dtype to str

    adata.obs.index = adata.obs.index.astype("str")

    return adata


def plot_metacell_umap(
    adata: AnnData, anno_col: str, min_dist: float = 0.5
) -> tuple[AnnData, plt.Axes]:
    """
    Plot metacells on UMAP, color by annotation column

    :param adata: an anndata object with metacells calculated by make_metacells
    :param anno_col: column in adata.obs with cell type annotation
    :param min_dist: minimum distance for UMAP calculation
    :return: a tuple of anndata object and a seaborn plot
    """

    global random_seed
    mc.pl.compute_umap_by_features(
        adata, max_top_feature_genes=1000, min_dist=min_dist, random_seed=random_seed
    )
    umap_x = mc.ut.get_o_numpy(adata, "umap_x")
    umap_y = mc.ut.get_o_numpy(adata, "umap_y")

    if anno_col not in adata.obs.columns:
        raise KeyError(f"{anno_col} is not in adata.obs.columns")

    if anno_col is not None:
        plot = sns.scatterplot(data=adata.obs, x=umap_x, y=umap_y, hue=anno_col)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    else:
        plot = sns.scatterplot(x=umap_x, y=umap_y)

    return adata, plot


def plot_counts_density(
    adata: AnnData,
) -> plt.Axes:
    """
    Plot density of total UMIs of cells

    :param adata: a raw count anndata object
    :return: a seaborn density plot
    """
    print("plot_counts_density")
    total_umis_of_cells = mc.ut.get_o_numpy(adata, name="__x__", sum=True)
    plot = sns.histplot(total_umis_of_cells, kde=True)
    plot.set(xlabel="UMIs", ylabel="Density", yticks=[])
    return plot


def plot_gene_similarity_module(
    adata: AnnData,
    module_num: int,
) -> plt.Axes:
    """
    Plot heatmap of gene similarity within a module
    :param adata: the anndata output by find_cell_cycle_gene_modules
    :param module_num: which module to plot
    :return: a seaborn heatmap plot
    """
    if "related_genes_module" not in adata.var.columns:
        raise KeyError(
            "related_genes_module is not in adata.var.columns, is gene modules computed by "
            "find_cell_cycle_gene_modules?"
        )

    similarity_of_genes = mc.ut.get_vv_frame(adata, "related_genes_similarity")
    module_of_genes = adata.var["related_genes_module"]

    module_genes_mask = module_of_genes == module_num
    similarity_of_module = similarity_of_genes.loc[module_genes_mask, module_genes_mask]

    similarity_of_module.index = similarity_of_module.columns = [
        name for name in similarity_of_module.index
    ]
    ax = plt.axes()
    sns.heatmap(
        similarity_of_module,
        vmin=0,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        ax=ax,
        cmap="YlGnBu",
    )
    ax.set_title(f"Gene Module {module_num}")
    return ax


def plot_num_sc_per_metacell(annotation_sc: pd.DataFrame) -> plt.Axes:
    """
    Plot histogram of number of single cells per metacell
    :param annotation_sc: dataframe with cell type annotation from collect_metacell_annotations
    :return: a seaborn histogram plot
    """
    plot = plt.axes()
    plt.hist(annotation_sc.query("metacell >= 0")["count"], bins=30)
    plt.title("Number of single cells per metacell")

    return plot


def plot_num_cell_type_sc_per_metacell(
    metacells_res: AnnData,
    anno_col: str,
) -> plt.Axes:
    """
    Plot histogram of number of cell types of single cells per metacell
    :param metacells_res: anndata object with metacells calculated by make_metacells
    :param anno_col: column in adata.obs with cell type annotation
    :return: a seaborn histogram plot
    """
    if "metacell" not in metacells_res.obs.columns:
        raise KeyError("metacell not in metacells_res.obs.columns")

    if anno_col not in metacells_res.obs.columns:
        raise KeyError(f"{anno_col} is not in adata.obs.columns")

    df = (
        metacells_res.obs[["metacell", anno_col]]
        .query("metacell >= 0")
        .groupby("metacell")[anno_col]
        .nunique()
    )

    plot = plt.axes()
    plt.hist(df, bins=10)
    plt.title("Number of cell types of single cells per metacell")

    return plot


def make_metacells_one_cell_type(
    adata: AnnData,
    cell_type_now: str,
    annotation_col: str,
    forbidden_gene_names: list[str],
    target_umi: int = 160000,
) -> AnnData:
    """
    Function to make metacells for one cell type only, used in make_metacells_per_group
    :param adata: anndata object
    :type adata: AnnData
    :param cell_type_now: name of cell type for which we are making metacell now
    :type cell_type_now: str
    :param annotation_col: column in adata.obs indicating cell type annotation
    :type annotation_col: str
    :param forbidden_gene_names: list of forbidden gene names, these genes are excluded in metacell construction
    :type forbidden_gene_names: list
    :param target_umi: target UMI counts per metacell, defaults to 160000
    :type target_umi: int, optional
    :return: anndata object with metacells for one cell type
    :rtype: AnnData
    """

    global random_seed
    adata_now = adata[adata.obs[annotation_col] == cell_type_now, :]

    max_parallel_piles = mc.pl.guess_max_parallel_piles(adata_now)
    print(f"max parallel piles for metacell calculation = {max_parallel_piles}")

    mc.pl.set_max_parallel_piles(max_parallel_piles)
    target_pile_size = mc.pl.compute_target_pile_size(
        adata_now, target_metacell_size=target_umi
    )

    print(f"target pile size = {target_pile_size}")

    # this function does not return anything
    mc.pl.divide_and_conquer_pipeline(
        adata_now,
        min_target_pile_size=5000,
        max_target_pile_size=12000,
        forbidden_gene_names=forbidden_gene_names,
        random_seed=random_seed,
    )
    adata_now_mc = mc.pl.collect_metacells(adata_now, name=cell_type_now)
    adata_now_mc.obs["cell_type"] = cell_type_now
    adata_now_mc.obs["cell_name"] = adata_now_mc.obs.apply(
        lambda row: row.cell_type + "_" + str(row.candidate), axis=1
    )
    adata_now_mc.obs_names = adata_now_mc.obs["cell_name"].values
    # sometimes there will be duplicated 'candidate', not sure why, but just to make the cell names informative I still use the cell_type+candidate as cell names
    adata_now_mc.obs_names_make_unique()

    return adata_now_mc


def make_metacells_per_group(
    adata: AnnData,
    annotation_col: str,
    forbidden_gene_names: list[str],
    target_umi: int = 160000,
) -> AnnData:
    """
    Make metacells for each cell type in adata, and combine them into one anndata object
    **Only used as an experimental function, not recommended for general use**
    :param adata: AnnData object
    :type adata: AnnData
    :param annotation_col: column in adata.obs indicating cell type annotation
    :type annotation_col: str
    :param forbidden_gene_names: list of forbidden gene names, these genes are excluded in metacell construction
    :type forbidden_gene_names: list
    :param target_umi: target UMI counts per metacell, defaults to 160000
    :type target_umi: int, optional
    :return: anndata object with metacells for each cell type
    :rtype: AnnData
    """
    all_cell_type_mc_adatas = list()
    all_feature_genes = list()
    for cell_type_now in adata.obs[annotation_col].unique():

        print(f"making metacells for {cell_type_now}")
        try:
            cell_type_now_mc_adata = make_metacells_one_cell_type(
                adata,
                cell_type_now,
                annotation_col,
                forbidden_gene_names,
                target_umi=target_umi,
            )

        except:
            print(
                f"making metacells for {cell_type_now} not successful, will create a summed cell instead"
            )
            cell_type_now_adata = adata[adata.obs[annotation_col] == cell_type_now, :]
            cell_type_now_summed = cell_type_now_adata.X.sum(axis=0)
            summed_obs = pd.DataFrame(
                {
                    "grouped": cell_type_now_adata.X.shape[
                        0
                    ],  # how many single cells in this metacell, int
                    "pile": 0,
                    "cell_type": cell_type_now,
                    "cell_name": cell_type_now,
                },
                index=[cell_type_now],
            )
            cell_type_now_summed_adata = sc.AnnData(
                X=np.array(cell_type_now_summed),
                obs=summed_obs,
                var=cell_type_now_adata.var,
            )
            cell_type_now_summed_adata.var["forbidden_gene"] = False
            cell_type_now_summed_adata.var["feature_gene"] = False
            all_cell_type_mc_adatas.append(cell_type_now_summed_adata)
        else:
            print(
                f"making metacells for {cell_type_now} successful, adding to total anndatas"
            )
            feature_genes_now = cell_type_now_mc_adata.var.loc[
                cell_type_now_mc_adata.var.feature_gene == 1
            ].index.values
            all_feature_genes.extend(feature_genes_now)
            all_cell_type_mc_adatas.append(cell_type_now_mc_adata)

    final_mc_adata = sc.concat(
        all_cell_type_mc_adatas, axis=0, join="outer", merge="same"
    )
    final_mc_adata.obs_names_make_unique()

    all_feature_genes = set(all_feature_genes)

    final_mc_adata.var["feature_gene"] = False
    final_mc_adata.var["forbidden_gene"] = False

    final_mc_adata.var.loc[
        final_mc_adata.var_names.isin(all_feature_genes), "feature_gene"
    ] = True
    final_mc_adata.var.loc[
        final_mc_adata.var_names.isin(forbidden_gene_names), "forbidden_gene"
    ] = True

    # for writing adata - str to cat can happen
    final_mc_adata.obs = final_mc_adata.obs.astype(str)
    final_mc_adata.var = final_mc_adata.var.astype(str)

    return final_mc_adata


class SummedAnnData(AnnData):
    def __init__(
        self,
        summed_adata=None,
        removed_genes: list[str] = None,
        removed_min_count: int = None,
        removed_min_cells_pct: float = None,
    ):
        super().__init__(
            X=summed_adata.X if summed_adata.X is not None else None,
            obs=summed_adata.obs if summed_adata.obs is not None else None,
            var=summed_adata.var if summed_adata.var is not None else None,
        )

        self._count_type = "summed_counts"
        self.removed_genes = removed_genes
        self.removed_min_count = removed_min_count
        self.removed_min_cells_pct = removed_min_cells_pct

    @property
    def count_type(self):
        return self._count_type

    @classmethod
    def create_from_summing_anndata(
        cls,
        adata: AnnData,
        annotation_col: str,
        removed_genes: list[str] = None,
        removed_min_count: int = None,
        removed_min_cells_pct: float = None,
    ):
        """create an SummedAnnData object from aggregating counts of cells of the same annotation in an AnnData objecy

        :param adata: anndata objecy
        :type adata: AnnData
        :param annotation_col: column in adata.obs indicating annotation to use
        :type annotation_col: str
        :param removed_genes: initiate for further filtering, defaults to None
        :type removed_genes: list, optional
        :param removed_min_cell_pct: initiate for further filtering, defaults to None
        :type removed_min_cell_pct: float, optional
        :param removed_min_count: initiate for further filtering, defaults to None
        :type removed_min_count: int, optional
        :return: a SummedAnnData object
        :rtype: SummedAnnData
        """
        summed_adata = sum_expression_by_class(
            adata=adata, annotation_col=annotation_col
        )
        return cls(
            summed_adata, removed_genes, removed_min_count, removed_min_cells_pct
        )

    @classmethod
    def create_from_metacells_anndata(
        cls,
        adata: AnnData,
        removed_genes: list[str] = None,
        removed_min_count: int = None,
        removed_min_cells_pct: float = None,
    ):
        """
        Make a SummedAnnData object from an AnnData object with metacells

        :param adata: anndata object with metacells
        :type adata: AnnData
        :param removed_genes: genes removed from further classification analysis, defaults to None
        :type removed_genes: list[str], optional
        :param removed_min_count: min_count for removed genes, defaults to None
        :type removed_min_count: int, optional
        :param removed_min_cells_pct: min percentage of cells expressed for removed genes, defaults to None
        :type removed_min_cells_pct: float, optional
        :return: a SummedAnnData object
        :rtype: SummedAnnData
        """
        return cls(adata, removed_genes, removed_min_count, removed_min_cells_pct)

    def depth_normalize_counts(self, target_sum: int = None):
        """Run depth normalization on SummedAnnData object

        :param target_sum: target sum for depth normalization, defaults to None, meaning average total counts across cells
        :type target_sum: int, optional
        :return: depth normalized SummedAnnData object
        :rtype: SummedAnnData
        """

        print("Size factor depth normalize counts")
        if target_sum is not None:
            print(f"Target total UMI per cell is {target_sum}")
        else:
            print(f"Target total UMI per cell is the average UMI across cells")
        sc.pp.normalize_total(self, target_sum=target_sum, inplace=True)
        print(f"Total UMI count is normalized to {self.X.sum(axis=1)[0].round()}")

        return self

    def filter_low_counts(self, min_count=1, min_cells_pct=0):
        """Remove genes that doesn't pass min_count, or doesn't have non-zero counts in min_cells_pct

        :param min_count: minimum count of gene, defaults to 1
        :type min_count: int, optional
        :param pct_detected: minimum percentage of expression across cells, defaults to 0
        :type pct_detected: int, optional
        :return: filtered SummedAnnData object with removed_genes, removed_min_count annotated
        :rtype: SummedAnnData
        """
        assert isinstance(
            min_count, (int, float)
        ), "min_count should be int or float type"
        assert isinstance(
            min_cells_pct, (int, float)
        ), "min_cells_pct should be int or float type"

        original_genes = (
            self.var_names.values
        )  # store the original list of genes for the record
        num_cells = len(self.obs_names)

        if min_count:
            print(f"Genes with min_count {min_count} are considered low count")
            sc.pp.filter_genes(self, min_counts=min_count, inplace=True)
        if min_cells_pct:
            print(f"Genes with min_cells_pct {min_cells_pct} are considered low count")
            min_cells = round(num_cells * min_cells_pct * 0.01)
            sc.pp.filter_genes(self, min_cells=min_cells, inplace=True)

        removed_genes = [x for x in original_genes if x not in self.var_names.values]
        self.removed_genes = removed_genes
        if min_count:
            self.removed_min_count = min_count
        if min_cells_pct:
            self.removed_min_cells_pct = min_cells_pct
        print(f"Put {len(removed_genes)} genes into low counts genes")
        return self


def sum_expression_by_class(adata, annotation_col):
    """
    Aggregate scRNA-seq anndata by cell class, create a "pseudo-bulk"
    :param adata: input anndata object
    :type adata: AnnData
    :param annotation_col: the column in adata.obs indicating cell groups to combine
    :type annotation_col: str
    :return: an anndata object storing combined counts for each cell group
    :rtype: AnnData
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
