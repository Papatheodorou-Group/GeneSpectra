"""
GO annotation of classified genes
Python 3.11.0
Yuyao Song <ysong@ebi.ac.uk>
Dec 2024
© EMBL-European Bioinformatics Institute, 2024
"""

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from biomart import BiomartServer

import os
import sys
from typing import Literal

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from classify_genes import GeneClassificationResult


def go_annotation_ensembl(
    data: GeneClassificationResult,
    species: str,
    go_type: Literal[
        "molecular_function", "biological_process", "cellular_component"
    ] = "molecular_function",
    go_linkage_type=(
        "experimental",
        "phylogenetic",
        "computational",
        "author",
        "curator",
    ),
) -> pd.DataFrame:
    """Get GO annotation for classified genes from biomart, choose which annotations to include

    :param data: the gene classification result
    :type data: GeneClassificationResult
    :param species: species name in ensembl format, e.g. "hsapiens", "mmusculus", "ggallus"
    :type species: str
    :param go_type: which GO to look for, defaults to "molecular_function"
    :type go_type: str, optional
    :param go_linkage_type: choose between GO annotations to include based on evidence code, defaults to ( "experimental", "phylogenetic", "computational", "author", "curator", )
    :type go_linkage_type: tuple, optional
    :return: the GO annotation table of genes included in the gene classification result
    :rtype: pd.DataFrame
    """

    # filter GO terms by evidence code
    go_source = {
        "experimental": [
            "EXP",
            "IDA",
            "IPI",
            "IMP",
            "IGI",
            "IEP",
            "HTP",
            "HDA",
            "HMP",
            "HGI",
            "HEP",
        ],
        "phylogenetic": ["IBA", "IBD", "IKR", "IRD"],
        "computational": ["ISS", "ISO", "ISA", "ISM", "IGC", "RCA"],
        "author": ["TAS", "NAS"],
        "curator": ["IC", "ND"],
        "electronic": ["IEA"],
    }

    # Connect to Biomart server
    server = BiomartServer("http://www.ensembl.org/biomart")

    columns = [
        "ensembl_gene_id",
        "external_gene_name",
        "go_id",
        "name_1006",
        "go_linkage_type",
        "namespace_1003",
    ]

    dataset = server.datasets[f"{species}_gene_ensembl"]

    print("fetching ENSEMBL annotation url...")

    response = dataset.search({"query": data.gene.tolist(), "attributes": columns})

    print("Downloading ENSEMBL annotation table...")

    eg2_go = pd.read_csv(response.url, sep="\t", header=None, names=columns)

    eg2_go = eg2_go[eg2_go["go_id"] != ""]

    go_terms = eg2_go.loc[eg2_go["namespace_1003"] == go_type]

    # remove the foundation term
    go_terms = go_terms[
        ~go_terms["name_1006"].isin(
            ["biological_process", "molecular_function", "cellular_component"]
        )
    ]

    types_in_source = {}

    for source_type in go_source:
        type_in = go_terms.loc[
            go_terms["go_linkage_type"].isin(go_source[source_type])
        ]["go_linkage_type"].unique()
        types_in_source[source_type] = type_in

    included_terms = [go_source[x] for x in go_linkage_type]
    print("including GO link types:")
    print(included_terms)

    use = [item for sublist in included_terms for item in sublist]

    go_terms = go_terms.loc[go_terms["go_linkage_type"].isin(use)]

    return go_terms


def plot_categories_go(
    data: GeneClassificationResult,
    go_terms: pd.DataFrame,
    go_query: str,
    column_name="external_gene_name",
    kind: Literal["spec_category", "dist_category"] = "spec_category",
) -> tuple[Figure, pd.DataFrame]:
    """

    :param data: the categories output by hpa_gene_classification
    :type data: GeneClassificationResult
    :param go_terms: the go terms table by go_annotation_ensembl
    :type go_terms: pd.DataFrame
    :param go_query: the query go term of interest, should be a GO id look like GO:0003700
    :type go_query: str
    :param kind: the kind of category to plot, choose between spec_category (default) and dist_category
    :type kind: str, optional
    :param column_name: the column name to match the 'gene' column in data, ensembl_gene_id or external_gene_name
    :return a plotly.graph_objects.Figure of Pie plot, and the data used for the plot
    :rtype: tuple[Figure, pd.DataFrame]
    """
    use_df = go_terms.loc[go_terms.go_id == go_query]
    counts_use = (
        data.loc[data.gene.isin(use_df[column_name])][kind]
        .value_counts()
        .to_frame()
        .reset_index()
    )
    fig = px.pie(
        counts_use,
        values=kind,
        names="index",
        title=f"Gene classes {use_df.name_1006.unique()[0]}",
    )
    fig.show()

    return fig, data.loc[data.gene.isin(use_df[column_name])][kind]
