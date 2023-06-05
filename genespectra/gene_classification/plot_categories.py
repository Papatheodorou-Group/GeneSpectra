import plotly.express as px
import pandas as pd


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