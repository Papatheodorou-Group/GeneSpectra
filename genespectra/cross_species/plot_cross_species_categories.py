"""
Plot gene class conservation heatmap
Python 3.11.0
Yuyao Song <ysong@ebi.ac.uk>
Dec 2024
© EMBL-European Bioinformatics Institute, 2024
"""

import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objects import Figure


def plot_cross_species_spec_category_heatmap(
    mapping_both: pd.DataFrame,
    species_1: str,
    species_2: str,
    group_enhanced: bool = False,
    title: str = "One2one orthologs",
) -> tuple[pd.DataFrame, Figure]:
    """Plot a heatmap to indicate cross-species gene SPECIFICITY class conservation pattern

    :param mapping_both: the orthology-mapped gene classes from two species
    :type mapping_both: pd.DataFrame
    :param species_1: first species name
    :type species_1: str
    :param species_2: second species name
    :type species_2: str
    :param group_enhanced: whether have group_enhanced as a separate class, or merge with cell_type_enhanced to become enhanced, sometimes when
    there are very few group enhanced genes it makes sense to merge, defaults to False
    :type group_enhanced: bool, optional
    :param title: title of the heatmap, defaults to "One2one orthologs"
    :type title: str, optional
    :return: a heatmap figure showing gene class conservation for each gene class pair between species
    :rtype: tuple[pd.DataFrame, Figure]
    """
    if not group_enhanced:
        mapping_both.loc[
            mapping_both[f"spec_category_{species_1}"] == "group enhanced",
            [f"spec_category_{species_1}"],
        ] = "enhanced"
        mapping_both.loc[
            mapping_both[f"spec_category_{species_1}"] == "cell type enhanced",
            [f"spec_category_{species_1}"],
        ] = "enhanced"

        mapping_both.loc[
            mapping_both[f"spec_category_{species_2}"] == "group enhanced",
            [f"spec_category_{species_2}"],
        ] = "enhanced"
        mapping_both.loc[
            mapping_both[f"spec_category_{species_2}"] == "cell type enhanced",
            [f"spec_category_{species_2}"],
        ] = "enhanced"

    df = (
        mapping_both.groupby([f"spec_category_{species_1}"])[
            f"spec_category_{species_2}"
        ]
        .value_counts()
        .to_frame("count")
        .reset_index()
    )

    species1_counts = df.groupby(f"spec_category_{species_1}")["count"].sum()
    species2_counts = df.groupby(f"spec_category_{species_2}")["count"].sum()

    df["Species1_Percentage"] = df.apply(
        lambda row: (row["count"] / species1_counts[row[f"spec_category_{species_1}"]])
        * 100,
        axis=1,
    )
    df["Species2_Percentage"] = df.apply(
        lambda row: (row["count"] / species2_counts[row[f"spec_category_{species_2}"]])
        * 100,
        axis=1,
    )

    # The heatmap shows the harmonic mean of the percentage of genes (orthologs) in each pair of gene class from both species
    # this is to account for the differences of total genes in each class from the two species

    df["Harmonic_Mean"] = 2 / (
        (1 / df["Species1_Percentage"]) + (1 / df["Species2_Percentage"])
    )

    heatmap_data = df.pivot(
        columns=f"spec_category_{species_1}",
        index=f"spec_category_{species_2}",
        values="Harmonic_Mean",
    )

    heatmap_data = np.round(heatmap_data.astype("float32"), 2)

    if group_enhanced and ("group enhanced" in heatmap_data.columns.values):
        order_columns = [
            "cell type enriched",
            "group enriched",
            "cell type enhanced",
            "group enhanced",
            "low cell type specificity",
            "lowly expressed",
        ]
    elif group_enhanced and not ("group enhanced" in heatmap_data.columns.values):
        # sometimes even when group_enhanced is wanted, there is no genes of this category in some of the species
        print(
            f"use group enhanced but there is no group enhanced genes in species {species_1}"
        )
        order_columns = [
            "cell type enriched",
            "group enriched",
            "cell type enhanced",
            "low cell type specificity",
            "lowly expressed",
        ]
    else:
        order_columns = [
            "cell type enriched",
            "group enriched",
            "enhanced",
            "low cell type specificity",
            "lowly expressed",
        ]

    if group_enhanced and ("group enhanced" in heatmap_data.index.values):
        order_index = [
            "cell type enriched",
            "group enriched",
            "cell type enhanced",
            "group enhanced",
            "low cell type specificity",
            "lowly expressed",
        ]
    elif group_enhanced and not ("group enhanced" in heatmap_data.index.values):
        print(
            f"use group enhanced but there is no group enhanced genes in species {species_2}"
        )
        order_index = [
            "cell type enriched",
            "group enriched",
            "cell type enhanced",
            "low cell type specificity",
            "lowly expressed",
        ]
    else:
        order_index = [
            "cell type enriched",
            "group enriched",
            "enhanced",
            "low cell type specificity",
            "lowly expressed",
        ]

    fig = px.imshow(
        heatmap_data[order_columns].reindex(order_index),
        title=title,
        text_auto=True,
        width=750,
        height=600,
        labels=dict(x=species_1, y=species_2, color="% overlap"),
    )
    fig["layout"]["yaxis"]["autorange"] = "reversed"
    fig.update_yaxes(autorange=True)

    return heatmap_data, fig


def plot_cross_species_dist_category_heatmap(
    mapping_both: pd.DataFrame,
    species_1: str,
    species_2: str,
    title="One2one orthologs",
) -> tuple[pd.DataFrame, Figure]:
    """Plot a heatmap to indicate cross-species gene DISTRIBUTION class conservation pattern

    :param mapping_both: the orthology-mapped gene classes from two species
    :type mapping_both: pd.DataFrame
    :param species_1: first species name
    :type species_1: str
    :param species_2: second species name
    :type species_2: str
    :param title: title of the heatmap, defaults to "One2one orthologs"
    :type title: str, optional
    :return: a heatmap figure showing gene class conservation for each gene class pair between species
    :rtype: tuple[pd.DataFrame, Figure]
    """

    df = (
        mapping_both.loc[
            mapping_both[f"{species_2}_homolog_orthology_type"] == "ortholog_one2one"
        ]
        .groupby([f"dist_category_{species_1}"])[f"dist_category_{species_2}"]
        .value_counts()
        .to_frame("count")
        .reset_index()
    )

    species1_counts = df.groupby(f"dist_category_{species_1}")["count"].sum()
    species2_counts = df.groupby(f"dist_category_{species_2}")["count"].sum()

    df["Species1_Percentage"] = df.apply(
        lambda row: (row["count"] / species1_counts[row[f"dist_category_{species_1}"]])
        * 100,
        axis=1,
    )
    df["Species2_Percentage"] = df.apply(
        lambda row: (row["count"] / species2_counts[row[f"dist_category_{species_2}"]])
        * 100,
        axis=1,
    )
    df["Harmonic_Mean"] = 2 / (
        (1 / df["Species1_Percentage"]) + (1 / df["Species2_Percentage"])
    )

    heatmap_data = df.pivot(
        columns=f"dist_category_{species_1}",
        index=f"dist_category_{species_2}",
        values="Harmonic_Mean",
    )

    heatmap_data = np.round(heatmap_data.astype("float32"), 2)

    order = [
        "expressed in over 90%",
        "expressed in over 30%",
        "expressed in less than 30%",
        "expressed in single",
        "lowly expressed",
    ]

    fig = px.imshow(
        heatmap_data[order].reindex(order),
        title=title,
        text_auto=True,
        width=750,
        height=600,
        labels=dict(x=species_1, y=species_2, color="% overlap"),
    )
    fig["layout"]["yaxis"]["autorange"] = "reversed"
    fig.update_yaxes(autorange=True)

    return heatmap_data, fig
