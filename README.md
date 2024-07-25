# Gene classification based on cell type expression specificity and distribution

The GeneSpectra module performs gene classification using scRNA-seq data. 

[Read our preprint here: Context-aware comparison of cell type gene expression across species](https://www.biorxiv.org/content/10.1101/2024.06.21.600109v1)

Steps:

1. Reduce sparsity by creating metacells or pooling
2. Normalize data and filter low count genes
3. Multi-thread gene classification for gene specificity and distribution
4. Compare orthologs classes between species

Note that the gene classes are modified based on Human Protein Atlas classifications by [Karlsson, M. et al.](https://www.science.org/doi/10.1126/sciadv.abh2169?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed})


## Modules

### Metacells

Wrapper functions and helper functions to use [metacells](https://github.com/tanaylab/metacells) to create metacells based on scRNA-seq data. It is also recommended to follow the official metacells workflow to create the most tailored metacells anndata object (use the [iterative vignette](https://tanaylab.github.io/metacells-vignettes/iterative.html) for brand-new data), as you have more freedom to adjust various parameters. Alternatively, when the dataset is unsuitable for metacell calculation, merge cells of the same annotation label to create cell pools.

### Gene classification

Core module to perform gene filtering, normalization, and gene specificity and distribution classification. Uses multi-processing to parallize the processing of genes.

### Cross species

Cross species comparison of gene classes, and plotting. Using [ensembl](https://www.ensembl.org/index.html) or [eggNOGV6](http://eggnog6.embl.de/) homology.

Developer / maintainer: Yuyao Song, <ysong@ebi.ac.uk>