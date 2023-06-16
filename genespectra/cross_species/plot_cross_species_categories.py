import pandas as pd
import numpy as np
import plotly.express as px


def plot_cross_species_category_heatmap(mapping_both, species_1, species_2):

    df = mapping_both.loc[mapping_both[f"{species_2}_homolog_orthology_type"] == 'ortholog_one2one'].groupby(
        [f"spec_category_{species_1}"])[f"spec_category_{species_2}"]\
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

    order = ['cell type enriched',
             'group enriched',
             'cell type enhanced',
             'group enhanced',
             'low cell type specificity',
             'lowly expressed']

    fig = px.imshow(heatmap_data[order].reindex(order),
                    title='One2one orthologs',
                    text_auto=True,
                    width=750, height=600,
                    labels=dict(x=species_2, y=species_1, color="% overlap"))
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.update_yaxes(autorange=True)
    fig.show()

    return fig

