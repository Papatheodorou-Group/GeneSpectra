# Gene classification based on cell type expression specificity and distribution

This module performs gene classification using scRNA-seq data. 

[Read our preprint here](https://www.biorxiv.org/content/10.1101/2024.06.21.600109v1)

Steps:

1. Reduce sparsity by creating metacells or pooling
2. Normalize data and filter low count genes
3. Multi-thread gene classification for gene specificity and distribution
4. Compare orthologs classes between species

Developer / maintainer: Yuyao Song, <ysong@ebi.ac.uk>
