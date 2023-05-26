import scanpy as sc
import numpy as np
import pandas as pd
import warnings
from numba import njit
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(rc={'figure.dpi': 100, 'figure.figsize': (2, 2)})


def remove_not_profiled_genes(adata):
    sc.pp.calculate_qc_metrics(adata, log1p=False, inplace=True)
    result_ad = sc.pp.filter_genes(adata, min_counts=1, copy=True)
    return result_ad


def depth_normalize_counts(adata):
    print("Size factor depth normalize counts")
    result_ad = sc.pp.normalize_total(adata, target_sum=None, copy=True)
    return result_ad


def log1p_counts(adata):
    print("Natural log1p counts data")
    result_ad = sc.pp.log1p(adata, copy=True)
    return result_ad


def find_low_count_genes(adata, min_count=1):
    print(f"find_low_count_genes, never above {min_count} in depth normalized counts across all metacells")
    columns = np.where(np.all(adata.X <= min_count, axis=0))[0]
    adata.var[f"never_above_{min_count}"] = False
    adata.var.iloc[columns, -1] = True
    print(adata.var[f"never_above_{min_count}"].value_counts())

    return adata


def remove_cell_cycle_genes(adata, cell_cycle_var_col='forbidden_gene'):

    result_ad = adata[:, ~adata.var[cell_cycle_var_col]].copy()
    return result_ad


def remove_lowly_expressed_genes(adata, min_count=1):
    result_ad = adata[:, ~adata.var[f"never_above_{min_count}"]].copy()

    return result_ad


def find_low_variance_genes(adata, var_cutoff=0.1):
    print("find_low variance_genes")
    if 'gene_var_log1psf' not in adata.var.columns:
        raise KeyError("gene_var_log1psf is not annotated for adata.var, run get_mean_var_disp first")
    adata.var['low_variance'] = adata.var['gene_var_log1psf'] <= var_cutoff
    print(adata.var['low_variance'].value_counts())

    if adata.var['low_variance'].sum() == adata.var.shape[0]:
        warnings.warn("All genes are low variance, consider lowering the var_cutoff")

    return adata


def find_low_expression_genes(adata, mean_cutoff=0.1):
    print("find_low expression_genes")
    if 'gene_mean_log1psf' not in adata.var.columns:
        raise KeyError("gene_mean_log1psf is not annotated for adata.var, run get_mean_var_disp first")
    adata.var['low_mean'] = adata.var['gene_mean_log1psf'] <= mean_cutoff
    print(adata.var['low_mean'].value_counts())

    if adata.var['low_mean'].sum() == adata.var.shape[0]:
        warnings.warn("All genes are low expression (mean), consider lowering the mean_cutoff")

    return adata


def subset_to_enhanced_genes():
    print("subset_to_enhanced_genes")


def get_mean_var_disp(adata, axis=0):
    X = choose_mtx_rep(adata, use_raw=False, layer=None)
    mean = np.mean(X, axis=axis, dtype=np.float64)
    mean_sq = np.multiply(X, X).mean(axis=axis, dtype=np.float64)
    var = mean_sq - mean ** 2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    mean[mean == 0] = 1e-12  # set entries equal to zero to small value
    dispersion = var / mean

    adata.var['gene_mean_log1psf'] = get_mean_var_disp(adata)[0]
    adata.var['gene_var_log1psf'] = get_mean_var_disp(adata)[1]
    adata.var['gene_dispersion_log1psf'] = get_mean_var_disp(adata)[2]

    return adata


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

@njit
def hpa_gene_classification(input_ad, anno_col, exp_lim, enr_fold, max_group_n):

    print("gene_classification")
    res = get_group_average(input_ad, anno_col)
    num_cell_types, num_genes = res.shape
    if max_group_n is None:
        max_group_n = np.floor(num_cell_types / 2)

    print(f"num cell types = {num_cell_types}, num_genes = {num_genes}, max_group_size = {max_group}")
    data = res.T.reset_index().melt(id_vars='gene', var_name='tissue', value_name='expression')
    data = pd.DataFrame(data)

    gene_col = 'gene'
    group_col = 'tissue'

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

    gene_class_info['lim'] = gene_class_info['max_exp'] / enr_fold
    gene_class_info['gene'] = gene_class_info.index

    gene_class_info['exps_over_lim'] = gene_class_info.apply(
        lambda row: list(
            data_.loc[
                (data_['gene'] == row['gene']) &
                (data_['expression'] >=row['lim']) &
                (data_['expression'] >= exp_lim)
                ]['expression']
        ),
        axis=1)

    # Enriched
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

    # Enhanced
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

    gene_class_info['enhanced_in'] = gene_class_info['exps_enhanced'].apply(
        lambda x: ';'.join(sorted(data_.loc[data_['expression'].isin(x)]['tissue'])))

    gene_class_info['spec_category'] = np.select(
        [
            gene_class_info['n_det'] == 0,
            gene_class_info['n_exp'] == 0,
            (gene_class_info['max_exp'] / gene_class_info['max_2nd_or_lim']) >= enr_fold,
            (gene_class_info['max_exp'] >= gene_class_info['lim']) &
            (gene_class_info['n_over'] <= max_group_n) &
            (gene_class_info['n_over'] > 1) &
            (gene_class_info['mean_over'] / gene_class_info['max_under_lim']) >= enr_fold,
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
    for clust in input_ad.obs['anno_col'].cat.categories:
        res.loc[clust] = input_ad[input_ad.obs[anno_col].isin([clust]), :].X.mean(0)
    return res
