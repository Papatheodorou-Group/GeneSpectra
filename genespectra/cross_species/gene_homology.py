"""
Gene homology mapping functions for comparing gene classification between species
Python 3.11.0
Yuyao Song <ysong@ebi.ac.uk>
Dec 2024
© EMBL-European Bioinformatics Institute, 2024
"""

import pandas as pd
from biomart import BiomartServer
from genespectra.gene_classification.classify_genes import GeneClassificationResult


def from_eggnog_to_ensembl_id(
    input_eggnog: pd.DataFrame, species_1: str, species_2: str
) -> pd.DataFrame:
    """
    For using eggNOG as homolgy method
    Function to map eggNog search of species_1 proteome (UniProt ID) to species_2 (ENSEMBL protein id), into ENSEMBL gene id for both species
    :param input_eggnog: a mapping table from out.emapper.orthologs (from running emapper), containing only the hits in the relevant species_2, use grep to get these entries
    :type input_eggnog: pd.DataFrame
    :param species_1: the ensembl name for species_1, such as mmusculus
    :type species_1: str
    :param species_2:the ensembl name for species_2, such as hsapiens
    :type species_2: str
    :return: a dataframe with protein ids mapped to ensembl gene id
    :rtype: pd.DataFrame
    """
    print("Map eggnog protein search orthologs to ensembl gene ids")
    mapping = input_eggnog

    if not all(mapping.columns == ["UniProt", "Type", "Species", "Other_id"]):
        KeyError(
            "column name of input_eggnog should be ['UniProt', 'Type', 'Species', 'Other_id'], please rename"
        )

    mapping["UniProt_id_sp1"] = mapping["UniProt"].apply(lambda x: x.split("|")[1])
    mapping["UniProt_id_source"] = mapping["UniProt"].apply(lambda x: x.split("|")[0])

    # use only swissprot entries as these proteins are experimentally verified
    mapping = mapping.loc[mapping.UniProt_id_source == "sp"]

    # Connect to the BioMart server
    server = BiomartServer("http://www.ensembl.org/biomart")
    # Select the Ensembl Gene ID dataset
    ensembl_sp1 = server.datasets[f"{species_1}_gene_ensembl"]
    # Define the columns for the query
    columns_sp1 = ["uniprot_gn_id", "ensembl_gene_id"]
    uniprot_ids_sp1 = mapping["UniProt_id_sp1"].tolist()
    # Query BioMart for the mappings of all UniProt IDs
    response_sp1 = ensembl_sp1.search(
        {
            "query": uniprot_ids_sp1,
            "attributes": columns_sp1,
            "mart_instance": "ensembl",
        }
    )
    mapping_df_sp1 = pd.read_csv(
        response_sp1.url, sep="\t", header=None, names=columns_sp1
    )
    mapping_df_sp1 = mapping_df_sp1.loc[mapping_df_sp1.uniprot_gn_id.notnull(), :]

    merged_df = pd.merge(
        mapping,
        mapping_df_sp1,
        left_on="UniProt_id_sp1",
        right_on="uniprot_gn_id",
        how="left",
    )

    # Drop the redundant columns
    merged_df.drop(columns=["uniprot_gn_id"], inplace=True)

    # Select the Ensembl Gene ID dataset
    ensembl_sp2 = server.datasets[f"{species_2}_gene_ensembl"]
    # Define the columns for the query
    columns_sp2 = ["ensembl_peptide_id", "ensembl_gene_id"]
    ensembl_prot_ids_sp1 = [x.replace("*", "") for x in mapping["Other_id"].tolist()]

    # Query BioMart for the mappings of all UniProt IDs
    response_sp2 = ensembl_sp2.search(
        {
            "query": ensembl_prot_ids_sp1,
            "attributes": columns_sp2,
            "mart_instance": "ensembl",
        }
    )

    # Convert the response to a dataframe
    mapping_df_sp2 = pd.read_csv(
        response_sp2.url, sep="\t", header=None, names=columns_sp2
    )
    mapping_df_sp2 = mapping_df_sp2.loc[mapping_df_sp2.ensembl_peptide_id.notnull(), :]
    merged_df["Other_id_clean"] = merged_df["Other_id"].str.replace("*", "")

    merged_df_sp2 = pd.merge(
        merged_df,
        mapping_df_sp2,
        left_on="Other_id_clean",
        right_on="ensembl_peptide_id",
        how="left",
        suffixes=("_sp1", "_sp2"),
    )

    # Drop the redundant columns
    merged_df_sp2.drop(columns=["ensembl_peptide_id"], inplace=True)

    merged_df_sp2 = merged_df_sp2.loc[
        merged_df_sp2.ensembl_gene_id_sp1.notnull()
        & merged_df_sp2.ensembl_gene_id_sp2.notnull(),
    ]

    return merged_df_sp2


# TODO: get code to match eggnog to gene classification results


def homology_mapping_ensembl(
    species_1: str,
    species_2: str,
    classes_sp1: GeneClassificationResult,
    classes_sp2: GeneClassificationResult,
    brief: bool = False,
) -> pd.DataFrame:
    """
    For using ENSEMBL as homology method
    Queries from ENSEMBL gene homology to map classified genes from two species using ENSEMBL gene id
    :param species_1: the ensembl name for species_1, such as mmusculus
    :type species_1: str
    :param species_2:the ensembl name for species_2, such as hsapiens
    :type species_2: str
    :param classes_sp1: the hpa classification from species 1
    :type classes_sp1: GeneClassificationResult
    :param classes_sp2: the hpa classification from species 2
    :type classes_sp2: GeneClassificationResult
    :param brief: whether return brief result or the full result
    :type brief: bool
    :return: a dataframe with gene classes from both species and genes mapped with ENSEMBL homology
    :rtype: pd.DataFrame
    """
    # Connect to the BioMart server
    server = BiomartServer("http://www.ensembl.org/biomart")
    # Select the Ensembl Gene ID dataset
    ensembl = server.datasets[f"{species_1}_gene_ensembl"]
    # Define the columns for the query
    columns = [
        "ensembl_gene_id",
        "external_gene_name",
        f"{species_2}_homolog_ensembl_gene",
        f"{species_2}_homolog_orthology_type",
        f"{species_2}_homolog_associated_gene_name",
    ]

    # Query BioMart for the mappings of orthologs genes
    print(f"total genes from {species_1}: {len(set(classes_sp1.gene))}")

    response = ensembl.search(
        {
            "query": classes_sp1.gene.tolist(),
            "attributes": columns,
            "mart_instance": "ensembl",
        }
    )

    mapping_df = pd.read_csv(response.url, sep="\t", header=None, names=columns)

    mapping_df = mapping_df.loc[
        mapping_df[f"{species_2}_homolog_ensembl_gene"].notnull()
    ]

    unique_genes_mapped_sp1 = len(
        list(set(classes_sp1.gene) & set(mapping_df.ensembl_gene_id))
    )

    print(f"ensembl homology mapped genes from {species_1}: {unique_genes_mapped_sp1}")

    if brief:

        minimum_classes_sp1 = classes_sp1[["gene", "spec_category", "enriched_tissues"]]
        minimum_classes_sp2 = classes_sp2[["gene", "spec_category", "enriched_tissues"]]
    else:
        minimum_classes_sp1 = classes_sp1
        minimum_classes_sp2 = classes_sp2

    mapping_sp1 = pd.merge(
        mapping_df,
        minimum_classes_sp1,
        left_on="ensembl_gene_id",
        right_on="gene",
        how="inner",
    )

    unique_genes_mapped_sp2 = len(
        list(
            set(minimum_classes_sp2.gene)
            & set(mapping_df[f"{species_2}_homolog_ensembl_gene"])
        )
    )

    print(f"total genes from {species_2}: {len(set(classes_sp2.gene))}")
    print(f"ensembl homology mapped genes from {species_2}: {unique_genes_mapped_sp2}")

    mapping_both = pd.merge(
        mapping_sp1,
        minimum_classes_sp2,
        left_on=f"{species_2}_homolog_ensembl_gene",
        right_on="gene",
        suffixes=(f"_{species_1}", f"_{species_2}"),
    ).drop(columns=["ensembl_gene_id", f"{species_2}_homolog_ensembl_gene"])

    return mapping_both
