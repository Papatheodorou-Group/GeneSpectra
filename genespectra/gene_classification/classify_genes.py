import scanpy as sc
import numpy as np
import pandas as pd
import warnings
from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing as mp

sns.set_theme(rc={'figure.dpi': 100, 'figure.figsize': (2, 2)})


def remove_not_profiled_genes(adata):
    """
    Filter genes that only have zeros across all cells i.e. at least one count
    :param adata: an anndata object
    :return: an anndata object with genes only have zeros removed
    """
    sc.pp.calculate_qc_metrics(adata, log1p=False, inplace=True)
    result_ad = sc.pp.filter_genes(adata, min_counts=1, copy=True)
    return result_ad


def depth_normalize_counts(adata, target_sum=None):
    """
    Wrapper of scanpy depth normalisation function
    Default use size-factor normalisation, size-factor is the average total count
    :param target_sum: target sum normalized count, default the average total count
    :param adata: an anndata object
    :return: an anndata object with normalized counts
    """
    print("Size factor depth normalize counts")
    if target_sum is not None:
        print(f"Target total UMI per cell is {target_sum}")
    else:
        print(f"Target total UMI per cell is the average UMI across cells")
    result_ad = sc.pp.normalize_total(adata, target_sum=target_sum, copy=True)
    print(f"Total UMI count is {result_ad.X.sum(axis=1)[0].round()}")

    return result_ad


def log1p_counts(adata):
    """
    Wrapper of scanpy log1p function, natural log transform normalized counts
    :param adata: an anndata object with normalized counts
    :return: an anndata object with log1p transformed counts
    """
    print("Natural log1p counts data")
    result_ad = sc.pp.log1p(adata, copy=True)
    return result_ad


def find_low_count_genes(adata, min_count=1):
    """
    Find genes never above min_count in all metacells, these genes are always lowly expressed
    :param adata: an anndata object with each cell a metacell
    :param min_count: minimum count, default 1
    :return: an anndata object with a boolean column in adata.var indicating whether the gene is lowly expressed
    """
    print(f"find_low_count_genes, never above {min_count} in depth normalized counts across all metacells")
    columns = np.where(np.all(adata.X <= min_count, axis=0))[0]
    adata.var[f"never_above_{min_count}"] = False
    adata.var.iloc[columns, -1] = True
    print(adata.var[f"never_above_{min_count}"].value_counts())

    return adata


def remove_cell_cycle_genes(adata, cell_cycle_var_col='forbidden_gene'):
    """
    Remove cell cycle genes and genes correlate with cell cycle genes found by metacell
    :param adata: an anndata object with metacell calculated
    :param cell_cycle_var_col: cell cycle gene column in adata.var, default 'forbidden_gene'
    :return: an anndata object with cell cycle and related genes removed
    """
    num_cell_cycle = adata[:, adata.var[cell_cycle_var_col]].shape[1]
    result_ad = adata[:, ~adata.var[cell_cycle_var_col]].copy()
    result_ad.uns['num_cell_cycle_genes'] = num_cell_cycle
    print(f"Removed {num_cell_cycle} cell cycle and related genes")
    return result_ad


def remove_low_counts_genes(adata, min_count=1):
    """
    Simply remove all lowly expressed genes found by find_low_count_genes
    :param adata: an anndata object with lowly expressed genes marked by find_low_count_genes
    :param min_count: the min_count used in find_low_count_genes
    :return: an anndata object with lowly expressed genes removed, and number of removed genes stored in .uns
    """
    num_lowly_expressed = adata[:, adata.var[f"never_above_{min_count}"]].shape[1]
    result_ad = adata[:, ~adata.var[f"never_above_{min_count}"]].copy()
    result_ad.uns['num_lowly_expressed'] = num_lowly_expressed
    print(f"Removed {num_lowly_expressed} low counts genes")

    return result_ad


def choose_mtx_rep(adata, use_raw=False, layer=None):
    is_layer = layer is not None
    if use_raw and is_layer:
        raise ValueError(
            "Cannot use expression from both layer and raw. You provided:"
            f"'use_raw={use_raw}' and 'layer={layer}'"
        )
    if is_layer:
        return adata.layers[layer]
    elif use_raw:
        return adata.raw.X
    else:
        return adata.X


def get_mean_var_disp(adata, axis=0):
    mat = choose_mtx_rep(adata, use_raw=False, layer=None)
    mean = np.mean(mat, axis=axis, dtype=np.float64)
    mean_sq = np.multiply(mat, mat).mean(axis=axis, dtype=np.float64)
    var = mean_sq - mean ** 2
    # enforce R convention (unbiased estimator) for variance
    var *= mat.shape[axis] / (mat.shape[axis] - 1)
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean
    adata.var['gene_mean_log1psf'] = mean
    adata.var['gene_var_log1psf'] = var
    adata.var['gene_dispersion_log1psf'] = dispersion

    return adata


def find_low_variance_genes(adata, var_cutoff=0.1):
    print("find_low variance_genes")
    if 'gene_var_log1psf' not in adata.var.columns:
        raise KeyError("gene_var_log1psf is not annotated for adata.var, run get_mean_var_disp first")
    adata.var['low_variance'] = adata.var['gene_var_log1psf'] <= var_cutoff
    print(adata.var['low_variance'].value_counts())

    if adata.var['low_variance'].sum() == adata.var.shape[0]:
        warnings.warn("All genes are low variance (var log1psf), consider lowering the var_cutoff")

    return adata


def find_low_expression_genes(adata, mean_cutoff=0.1):
    print("find_low expression_genes")
    if 'gene_mean_log1psf' not in adata.var.columns:
        raise KeyError("gene_mean_log1psf is not annotated for adata.var, run get_mean_var_disp first")
    adata.var['low_mean'] = adata.var['gene_mean_log1psf'] <= mean_cutoff
    print(adata.var['low_mean'].value_counts())

    if adata.var['low_mean'].sum() == adata.var.shape[0]:
        warnings.warn("All genes are low expression (mean log1psf), consider lowering the mean_cutoff")

    return adata


def subset_to_enhanced_genes():
    print("subset_to_enhanced_genes")


def plot_mean_var(adata, mean_ref, var_ref):
    ax = plt.axes()
    sns.scatterplot(data=adata.var, x="gene_mean_log1psf", y="gene_var_log1psf", size=1)
    if mean_ref is not None:
        plt.axvline(x=mean_ref, color='green', linestyle='--')
    if var_ref is not None:
        plt.axhline(y=var_ref, color='green', linestyle='--')
    return ax


def plot_mean_var_pie(adata):
    if 'low_variance' not in adata.var.columns or 'low_mean' not in adata.var.columns:
        raise KeyError("low variance and low mean not in adata.var.columns"
                       "run find_low_variance_genes and find_low_expression_genes on adata")

    low_mean_low_var = adata.var.loc[adata.var.low_variance & adata.var.low_mean, :].shape[0]
    high_mean_low_var = adata.var.loc[adata.var.low_variance & ~adata.var.low_mean, :].shape[0]
    high_mean_high_var = adata.var.loc[~adata.var.low_variance & ~adata.var.low_mean, :].shape[0]
    low_mean_high_var = adata.var.loc[~adata.var.low_variance & adata.var.low_mean, :].shape[0]

    y = np.array([high_mean_high_var, high_mean_low_var, low_mean_high_var, low_mean_low_var])
    labels = ["high_mean_high_var", "high_mean_low_var", "low_mean_high_var", "low_mean_low_var"]
    ax = plt.axes()
    plt.pie(y,
            labels=labels,
            autopct='%.1f%%',
            wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'},
            textprops={'size': 'medium'})
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=2)
    return ax


def prepare_anndata_for_classification(input_ad, anno_col):
    """
    :param input_ad: an anndata with each cell a metacell, size-factor normalized but not log transformed
    :param anno_col: the column in adata.obs with cell groups information, usually cell type
    :return:
    """
    res = get_group_average(input_ad, anno_col)
    data = res.T.reset_index().rename(columns={"index": "gene"}).melt(id_vars='gene', var_name='tissue',
                                                                      value_name='expression')
    data = pd.DataFrame(data)

    return data


def hpa_gene_classification(data: pd.DataFrame,
                            max_group_n: int = None,
                            exp_lim: float = 1,
                            enr_fold: float = 5
                            ) -> pd.DataFrame:
    """
    Core function to run HPA classification of genes, all genes are classified into:
    - Not detected: expression value never above zero
    - Lowly expressed: expression value never above exp_lim
    - Cell type enhanced: in one cell type the expression value is enr_fold the dataset average expression value
    - Group enhanced: in a few cell types, the average expression value is enr_fold the dataset average expression value
    - Cell type enriched: in one cell type the expression value is enr_fold all the other cell types
    - Group enriched: in a few cell types, the average expression value is enr_fold all the other cell types
    - Low cell type specificity: none of the above

    Number of expressed cell types are also reported
    :param data:
    :param exp_lim: the limit of expression, default 1
    :param enr_fold: the fold for enrichment and enhancement
    :param max_group_n: maximum number of cell types for group enrichment and enhancement, default half of all groups
    :return: a pd.dataframe containing information and classification of all genes
    """
    print("Running HPA gene classification \n")

    gene_col = 'gene'
    group_col = 'tissue'

    # by default, max groups is at most 50% of the number of groups

    if max_group_n is None:
        max_group_n = np.floor(len(data['tissue'].astype('category').cat.categories) / 2)

    num_cell_types = len(data['tissue'].astype('category').cat.categories)
    num_genes = len(data['gene'].astype('category').cat.categories)
    print(f"num cell types = {num_cell_types}, num_genes = {num_genes}, max_group={max_group_n}\n")

    data['expression'] = np.round(data['expression'].astype("float32"), 4)

    if data['expression'].isna().any():
        raise ValueError("NAs in expression column")
    if data[gene_col].isna().any():
        raise ValueError("NAs in gene column")
    if data[group_col].isna().any():
        raise ValueError("NAs in group column")

    data_ = data

    gene_class_info = data_.groupby('gene').agg(
        mean_exp=('expression', np.mean),
        min_exp=('expression', np.min),
        max_exp=('expression', np.max),
        max_2nd=('expression', lambda x: np.sort(x)[len(x) - 2])
    )

    # Expression frequency metrics
    gene_class_info['n_det'] = data_.groupby('gene')['expression'].apply(lambda x: np.sum(x > 0))
    gene_class_info['n_exp'] = data_.groupby('gene')['expression'].apply(lambda x: np.sum(x >= exp_lim))
    gene_class_info['frac_exp'] = gene_class_info['n_exp'] / data_.groupby('gene')['expression'].count() * 100
    gene_class_info['tissues_detected'] = data_.loc[data_['expression'] >= exp_lim].groupby('gene')['tissue'].apply(
        lambda x: ';'.join(sorted(x)))
    gene_class_info['tissues_not_detected'] = data_.loc[data_['expression'] < exp_lim].groupby('gene')['tissue'].apply(
        lambda x: ';'.join(sorted(x)))

    # enrichment limit
    gene_class_info['lim'] = gene_class_info['max_exp'] / enr_fold
    gene_class_info['gene'] = gene_class_info.index

    gene_class_info['exps_over_lim'] = gene_class_info.apply(
        lambda row: list(
            data_.loc[
                (data_['gene'] == row['gene']) &
                (data_['expression'] >= row['lim']) &
                (data_['expression'] >= exp_lim)
                ]['expression']
        ),
        axis=1)

    # Enriched genes
    # single cell type or group average enr_fold larger than any other cell type
    gene_class_info['n_over'] = gene_class_info['exps_over_lim'].apply(lambda x: len(x))
    gene_class_info['mean_over'] = gene_class_info['exps_over_lim'].apply(lambda x: np.mean(x))
    gene_class_info['min_over'] = gene_class_info.apply(
        lambda row: np.min(row['exps_over_lim']) if row['n_over'] > 0 else np.nan, axis=1)

    gene_class_info['max_under_lim'] = gene_class_info.apply(
        lambda row: np.maximum(
            np.max(
                data_.loc[
                    (data_['gene'] == row['gene']) &
                    (data_['expression'] < row['min_over'])
                    ]['expression']
            ),
            exp_lim * 0.1
        ),
        axis=1)

    gene_class_info['enrichment_group'] = gene_class_info.apply(
        lambda row: ';'.join(
            sorted(
                data_.loc[
                    (data_['gene'] == row['gene']) &
                    (data_['expression'] >= row['lim']) &
                    (data_['expression'] >= exp_lim)]
                ['tissue']
            )
        ),
        axis=1)

    gene_class_info['n_enriched'] = gene_class_info.apply(
        lambda row: data_.loc[
            (data_['gene'] == row['gene']) &
            (data_['expression'] >= row['lim']) &
            (data_['expression'] >= exp_lim)]
        ['tissue']
        .count(),
        axis=1)

    gene_class_info['max_2nd_or_lim'] = gene_class_info[['max_2nd', 'lim']].max(axis=1)

    # Enhanced genes
    # in some cell types, expression value over enr_fold the mean expression
    gene_class_info['exps_enhanced'] = gene_class_info.apply(
        lambda row: list(
            data_.loc[
                (data_['gene'] == row['gene']) &
                (data_['expression'] / row['mean_exp'] >= enr_fold) &
                (data_['expression'] >= exp_lim)
                ]['expression']
        ),
        axis=1)
    gene_class_info['n_enhanced'] = gene_class_info['exps_enhanced'].apply(lambda x: len(x))

    gene_class_info['enhanced_in'] = gene_class_info['enhanced_in'] = gene_class_info.apply(
        lambda row: ';'.join(
            sorted(
                data_.loc[
                    (data_['gene'] == row['gene']) &
                    (data_['expression'] / row['mean_exp'] >= enr_fold) &
                    (data_['expression'] >= exp_lim)
                    ]['tissue']
            )
        ),
        axis=1)
    # assign gene categories with this order
    # first satisfied condition option is used when multiple are satisfied
    gene_class_info['spec_category'] = np.select(
        [
            gene_class_info['n_det'] == 0,
            gene_class_info['n_exp'] == 0,
            (gene_class_info['max_exp'] / gene_class_info['max_2nd_or_lim']) >= enr_fold,
            (gene_class_info['max_exp'] >= gene_class_info['lim']) & (gene_class_info['n_over'] <= max_group_n) & (
                    gene_class_info['n_over'] > 1) & (
                    (gene_class_info['mean_over'] / gene_class_info['max_under_lim']) >= enr_fold),
            gene_class_info['n_enhanced'] == 1,
            gene_class_info['n_enhanced'] > 1
        ],
        [
            "not detected",
            "lowly expressed",
            "cell type enriched",
            "group enriched",
            "cell type enhanced",
            "group enhanced"
        ],
        default="low cell type specificity"
    )

    # Dist category
    gene_class_info['dist_category'] = np.select(
        [
            gene_class_info['frac_exp'] == 100,
            gene_class_info['frac_exp'] >= 31,
            gene_class_info['n_exp'] > 1,
            gene_class_info['n_exp'] == 1,
            gene_class_info['n_exp'] == 0,
            gene_class_info['n_det'] == 0
        ],
        [
            "expressed in all",
            "expressed in many",
            "expressed in some",
            "expressed in single",
            "lowly expressed",
            "not detected"
        ],
        default=""
    )

    # Spec score
    gene_class_info['spec_score'] = np.select(
        [
            gene_class_info['spec_category'] == "cell type enriched",
            gene_class_info['spec_category'] == "group enriched",
            gene_class_info['spec_category'].isin(["cell type enhanced", "group enhanced"])
        ],
        [
            gene_class_info['max_exp'] / gene_class_info['max_2nd_or_lim'],
            gene_class_info['mean_over'] / gene_class_info['max_under_lim'],
            gene_class_info['max_exp'] / gene_class_info['mean_exp']
        ],
        default=0.0
    )

    result = gene_class_info.assign(
        enriched_tissues=np.select(
            [
                gene_class_info['spec_category'].isin(["cell type enriched", "group enriched"]),
                gene_class_info['spec_category'].isin(["cell type enhanced", "group enhanced"])
            ],
            [
                gene_class_info['enrichment_group'],
                gene_class_info['enhanced_in']
            ],
            default=""
        ),
        n_enriched=np.select(
            [
                gene_class_info['spec_category'].isin(["cell type enriched", "group enriched"]),
                gene_class_info['spec_category'].isin(["cell type enhanced", "group enhanced"])
            ],
            [
                gene_class_info['n_enriched'],
                gene_class_info['n_enhanced']
            ],
            default=0
        )
    )

    return result


def get_group_average(input_ad, anno_col):
    input_ad.obs[anno_col] = input_ad.obs[anno_col].astype('category')
    res = pd.DataFrame(columns=input_ad.var_names, index=input_ad.obs[anno_col].cat.categories)
    for clust in input_ad.obs[anno_col].cat.categories:
        res.loc[clust] = input_ad[input_ad.obs[anno_col].isin([clust]), :].X.mean(0)
    return res


def batch_dataframe(data, num_gene_batches, random_selection=False, random_seed=123):
    if random_selection:
        np.random.seed(random_seed)

    unique_genes = data['gene'].unique()

    if random_selection:
        np.random.shuffle(unique_genes)

    gene_batches = np.random.choice(range(num_gene_batches), len(unique_genes))

    gene_batch_mapping = dict(zip(unique_genes, gene_batches))

    data['gene_batch'] = data['gene'].map(gene_batch_mapping)

    return data


# Define the function that will be applied to each group
def process_group(group, max_group_n=None, exp_lim=0.01, enr_fold=4):
    processed_data = hpa_gene_classification(group, max_group_n=max_group_n, exp_lim=exp_lim, enr_fold=enr_fold)

    return processed_data


# Your original function that operates on the grouped DataFrame
def hpa_gene_classification_multiprocess(data, num_gene_batches=10, random_selection=False, random_seed=123,
                                         max_group_n=None, exp_lim=0.01, enr_fold=4):
    """
    Multiprocessing to speed up HPA gene classification function
    Split the genes into num_gene_batches and process them in parallel
    Very helpful for large datasets
    :param random_seed: numpy random seed
    :param num_gene_batches: number of batches to group all genes, default 10, takes a few sec to run for 32k genes
    :param random_selection: whether randomly shuffle the genes when batching
    :param data: the input data from prepare_anndata_for_classification
    :param exp_lim: the limit of expression, default 1
    :param enr_fold: the fold for enrichment and enhancement
    :param max_group_n: maximum number of cell types for group enrichment and enhancement, default half of all groups
    :return: a pd.dataframe containing information and classification of all genes
    """

    # Create a pool of workers
    pool = mp.Pool(mp.cpu_count())

    # Batch and split the DataFrame into groups based on the specified column
    df = batch_dataframe(data, num_gene_batches=num_gene_batches, random_selection=random_selection, random_seed=random_seed)

    groups = df.groupby('gene_batch')

    # Apply the function to each group in parallel
    results = pool.starmap(process_group, [(group, max_group_n, exp_lim, enr_fold) for name, group in groups])

    # Close the pool of workers
    pool.close()
    pool.join()

    return pd.concat(results)


