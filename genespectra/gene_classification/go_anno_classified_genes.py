import pandas as pd
import plotly.express as px
from biomart import BiomartServer

import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from classify_genes import GeneClassificationResult


def go_annotation_ensembl(data: GeneClassificationResult, species,
                          go_type="molecular_function",
                          go_linkage_type=("experimental", "phylogenetic", "computational", "author", "curator")):
    go_source = {
        "experimental": ["EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI", "HEP"],
        "phylogenetic": ["IBA", "IBD", "IKR", "IRD"],
        "computational": ["ISS", "ISO", "ISA", "ISM", "IGC", "RCA"],
        "author": ["TAS", "NAS"],
        "curator": ["IC", "ND"],
        "electronic": ["IEA"]
    }

    # Connect to Biomart server
    server = BiomartServer("http://www.ensembl.org/biomart")

    columns = ["ensembl_gene_id", "external_gene_name", "go_id", "name_1006", "go_linkage_type", "namespace_1003"]

    dataset = server.datasets[f"{species}_gene_ensembl"]

    print("fetching ENSEMBL annotation url...")

    response = dataset.search({
        "query": data.gene.tolist(),
        "attributes": columns
    })

    print("Downloading ENSEMBL annotation table...")

    eg2_go = pd.read_csv(response.url, sep='\t', header=None, names=columns)

    eg2_go = eg2_go[eg2_go['go_id'] != ""]

    go_terms = eg2_go.loc[eg2_go['namespace_1003'] == go_type]

    # remove the foundation term
    go_terms = go_terms[~go_terms['name_1006'].isin(["biological_process", "molecular_function", "cellular_component"])]

    types_in_source = {}

    for source_type in go_source:
        type_in = go_terms.loc[go_terms['go_linkage_type'].isin(go_source[source_type])]['go_linkage_type'].unique()
        types_in_source[source_type] = type_in

    included_terms = [go_source[x] for x in go_linkage_type]
    print("including GO link types:")
    print(included_terms)

    use = [item for sublist in included_terms for item in sublist]

    go_terms = go_terms.loc[go_terms['go_linkage_type'].isin(use)]

    return go_terms


def plot_categories_go(data: GeneClassificationResult,
                       go_terms,
                       go_query,
                       column_name='external_gene_name'):
    use_df = go_terms.loc[go_terms.go_id == go_query]
    counts_use = data.loc[data.gene.isin(use_df[column_name])][
        'spec_category'].value_counts().to_frame().reset_index()
    fig = px.pie(counts_use, values='spec_category', names='index',
                 title=f"Gene classes {use_df.name_1006.unique()[0]}")
    fig.show()

    return fig
