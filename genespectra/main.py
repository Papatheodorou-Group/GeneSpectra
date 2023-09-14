import scanpy as sc

import os

directory_of_this_script = os.path.dirname(os.path.abspath(__file__))
print(directory_of_this_script)

from genespectra.gene_classification.classify_genes import remove_not_profiled_genes
from genespectra.gene_classification.classify_genes import remove_cell_cycle_genes
from genespectra.gene_classification.classify_genes import remove_low_counts_genes
from genespectra.gene_classification.classify_genes import depth_normalize_counts
from genespectra.gene_classification.classify_genes import find_low_count_genes
from genespectra.gene_classification.classify_genes import hpa_gene_classification



ad = sc.read_h5ad("/nfs/research/irene/ysong/RESULTS/GeneCellType/SCCAF_clustering_opt/Hearts/metacell_topics"
                  "/hs_heart_metacells_anno.h5ad")
ad = remove_not_profiled_genes(ad)
ad = depth_normalize_counts(ad)
ad = find_low_count_genes(ad, min_count=1)
ad = remove_low_counts_genes(ad)
ad = remove_cell_cycle_genes(ad)

result = hpa_gene_classification(ad, anno_col='cell_ontology_base', max_group_n=None, exp_lim=1, enr_fold=4)
