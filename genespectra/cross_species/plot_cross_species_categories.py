import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure


def plot_cross_species_spec_category_heatmap(mapping_both, species_1, species_2, group_enhanced=False) -> Figure:
    if not group_enhanced:
        mapping_both.loc[
            mapping_both[f"spec_category_{species_1}"] == 'group enhanced', [f"spec_category_{species_1}"]] = 'enhanced'
        mapping_both.loc[mapping_both[f"spec_category_{species_1}"] == 'cell type enhanced', [
            f"spec_category_{species_1}"]] = 'enhanced'

        mapping_both.loc[
            mapping_both[f"spec_category_{species_2}"] == 'group enhanced', [f"spec_category_{species_2}"]] = 'enhanced'
        mapping_both.loc[mapping_both[f"spec_category_{species_2}"] == 'cell type enhanced', [
            f"spec_category_{species_2}"]] = 'enhanced'

    df = mapping_both.loc[mapping_both[f"{species_2}_homolog_orthology_type"] == 'ortholog_one2one'].groupby(
        [f"spec_category_{species_1}"])[f"spec_category_{species_2}"] \
        .value_counts().to_frame().reset_index()

    species1_counts = df.groupby(f"spec_category_{species_1}")['count'].sum()
    species2_counts = df.groupby(f"spec_category_{species_2}")['count'].sum()

    df['Species1_Percentage'] = df.apply(
        lambda row:
        (row['count'] / species1_counts[row[f"spec_category_{species_1}"]]) * 100,
        axis=1)
    df['Species2_Percentage'] = df.apply(
        lambda row:
        (row['count'] / species2_counts[row[f"spec_category_{species_2}"]]) * 100,
        axis=1)
    df['Harmonic_Mean'] = 2 / ((1 / df['Species1_Percentage']) + (1 / df['Species2_Percentage']))

    heatmap_data = df.pivot(columns=f"spec_category_{species_1}",
                            index=f"spec_category_{species_2}",
                            values='Harmonic_Mean')

    heatmap_data = np.round(heatmap_data.astype("float32"), 4)

    if not group_enhanced:

        order = ['cell type enriched',
                 'group enriched',
                 'enhanced',
                 'low cell type specificity',
                 'lowly expressed']
    else:
        order = ['cell type enriched',
                 'group enriched',
                 'cell type enhanced',
                 'group enhanced',
                 'low cell type specificity',
                 'lowly expressed']

    fig = px.imshow(heatmap_data[order].reindex(order), title='One2one orthologs',
                    text_auto=True,
                    width=750, height=600,
                    labels=dict(x=species_2, y=species_1, color="% overlap"))
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_yaxes(autorange=True)
    fig.show()

    return fig


def plot_cross_species_dist_category_heatmap(mapping_both, species_1, species_2) -> Figure:
    df = mapping_both.loc[mapping_both[f"{species_2}_homolog_orthology_type"] == 'ortholog_one2one'].groupby(
        [f"dist_category_{species_1}"])[f"dist_category_{species_2}"] \
        .value_counts().to_frame().reset_index()

    species1_counts = df.groupby(f"dist_category_{species_1}")['count'].sum()
    species2_counts = df.groupby(f"dist_category_{species_2}")['count'].sum()

    df['Species1_Percentage'] = df.apply(
        lambda row:
        (row['count'] / species1_counts[row[f"dist_category_{species_1}"]]) * 100,
        axis=1)
    df['Species2_Percentage'] = df.apply(
        lambda row:
        (row['count'] / species2_counts[row[f"dist_category_{species_2}"]]) * 100,
        axis=1)
    df['Harmonic_Mean'] = 2 / ((1 / df['Species1_Percentage']) + (1 / df['Species2_Percentage']))

    heatmap_data = df.pivot(columns=f"dist_category_{species_1}",
                            index=f"dist_category_{species_2}",
                            values='Harmonic_Mean')

    heatmap_data = np.round(heatmap_data.astype("float32"), 4)

    order = ['expressed in over 90%',
             'expressed in over 50%',
             'expressed in over 25%',
             'expressed in less than 25%',
             'expressed in single',
             'lowly expressed']

    fig = px.imshow(heatmap_data[order].reindex(order), title='One2one orthologs',
                    text_auto=True,
                    width=750, height=600,
                    labels=dict(x=species_2, y=species_1, color="% overlap"))
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_yaxes(autorange=True)
    fig.show()

    return fig
