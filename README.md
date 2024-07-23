# Gene classification based on cell type expression specificity and distribution

This module performs gene classification using scRNA-seq data. 

[Read our preprint here](https://www.biorxiv.org/content/10.1101/2024.06.21.600109v1)

Steps:

1. Reduce sparsity by creating metacells or pooling
2. Normalize data and filter low count genes
3. Multi-thread gene classification for gene specificity and distribution
4. Compare orthologs classes between species

Note that the gene classes are modified based on Human Protein Atlas classifications by [Karlsson, M. et al.](https://www.science.org/doi/10.1126/sciadv.abh2169?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed})

Developer / maintainer: Yuyao Song, <ysong@ebi.ac.uk>
