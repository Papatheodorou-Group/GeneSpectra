import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(rc={'figure.dpi': 100, 'figure.figsize': (2, 2)})
import pandas as pd


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


def plot_categories_pie(data):
    """

    :param data: the categories output by hpa_gene_classification
    :return: a plotly.graph_objects.Figure
    """
    counts = data['spec_category'].value_counts().to_frame().reset_index().rename(
        columns={"spec_category": "counts", "index": 'gene_category'})
    fig = px.pie(counts, values='counts', names='gene_category', title='Human heart')
    fig.show()

    return fig


def plot_categories_hist(data):
    fig = px.histogram(data, x='n_exp', color='spec_category', nbins=30,
                       opacity=0.8, barmode='stack')

    fig.show()
    return fig
