import scanpy as sc
from genespectra.gene_classification.classify_genes import ExpressionDataLong
from genespectra.gene_classification.classify_genes import depth_normalize_counts
from genespectra.gene_classification.classify_genes import gene_classification_multiprocess
from genespectra.metacells.make_metacells import sum_expression_by_class
import click


@click.command()
@click.argument("input_h5ad", type=click.Path(exists=True))
@click.argument("out_gene_class", type=click.Path(exists=False), default=None)
@click.option('--anno_col', type=str, default=None, help="Cell class annotation column to use in input_h5ad")
def run_classification_cell_pool(input_h5ad, out_gene_class, anno_col, **kwargs):
    adata = sc.read_h5ad(input_h5ad)
    summed_adata = sum_expression_by_class(adata, anno_col)

    # when running on sum data, since different cell type have a different number of cells,
    # normalize to a fixed size factor
    ad_norm = depth_normalize_counts(summed_adata, target_sum=1000000)
    data = ExpressionDataLong()
    data = data.create_from_adata(input_ad=ad_norm, anno_col=anno_col)
    result_classes = gene_classification_multiprocess(data, **kwargs)
    result_classes.to_csv(out_gene_class)


if __name__ == '__main__':
    run_classification_cell_pool()
