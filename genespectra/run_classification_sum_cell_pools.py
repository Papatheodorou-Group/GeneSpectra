#!/usr/bin/env Python3

"""
Example script for running gene classification on cell type-summed anndata using genespectra
If metacells are used as input, it is recommended to build metacells beforehand or refer to the metacells submodule here
It is better to adjust the metacell construction process interactively in a jupyter notebook
Python 3.11.0
Yuyao Song <ysong@ebi.ac.uk>
Dec 2024
© EMBL-European Bioinformatics Institute, 2024
"""

import scanpy as sc
from genespectra.gene_classification.classify_genes import (
    ExpressionDataLong,
    GeneClassificationResult,
)
from genespectra.metacells.make_metacells import SummedAnnData
import click


@click.command()
@click.argument("input_h5ad", type=click.Path(exists=True))
@click.argument("out_gene_class", type=click.Path(exists=False), default=None)
@click.option(
    "--anno_col",
    type=str,
    default=None,
    help="Cell class annotation column to use in input_h5ad",
)
def run_classification_cell_pool(
    input_h5ad, out_gene_class, anno_col, target_sum=1000000, **kwargs
):
    """function to run gene classification for pseudobulk data

    :param input_h5ad: input file path, .h5ad file neeeded
    :type input_h5ad: str
    :param out_gene_class: output gene classes table file path, csv
    :type out_gene_class: str
    :param anno_col: column name for cell type annotation
    :type anno_col: str
    :param target_sum: target sum for depth normalization
    :type target_sum: int
    """
    adata = sc.read_h5ad(input_h5ad)

    # simply make pseudobulks, sum by anno_col
    summed_adata = SummedAnnData.create_from_anndata(adata, annotation_col=anno_col)
    print(summed_adata)

    # normalize to a fixed size factor
    summed_adata = SummedAnnData.depth_normalize_counts(
        summed_adata, target_sum=target_sum
    )
    sc.pp.calculate_qc_metrics(summed_adata, log1p=False, inplace=True)
    summed_adata = SummedAnnData.filter_low_counts(summed_adata, min_count=1)

    print(f"running gene classification on {len(summed_adata.var_names.values)} genes")
    expr_data = ExpressionDataLong.create_from_summed_adata(
        input_summed_adata=summed_adata, anno_col=anno_col
    )
    result_classes = (
        GeneClassificationResult.create_from_expression_data_long_multiprocess(
            expr_data, **kwargs
        )
    )
    result_classes.to_csv(out_gene_class)


if __name__ == "__main__":
    run_classification_cell_pool()
