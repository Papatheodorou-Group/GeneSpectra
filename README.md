# Gene classification based on cell type expression specificity and distribution

The GeneSpectra module performs gene classification using scRNA-seq data. 

[Read our preprint here: Revising the ortholog conjecture in cross-species comparison of scRNA-seq data (v3)](https://www.biorxiv.org/content/10.1101/2024.06.21.600109v3)

Analysis steps provided in this package:

1. Reduce sparsity by creating metacells or pseudobulking
2. Normalise data and filter low-count genes
3. Multi-thread gene classification for gene specificity and distribution
4. Visualisation of gene classification results
5. Compare ortholog classes between species and generate the gene class conservation heatmap

Note that the gene classes are modified based on Human Protein Atlas classifications by [Karlsson, M. et al.](https://www.science.org/doi/10.1126/sciadv.abh2169?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed})

## Install

First pull source code from the repository:

```shell
git clone https://github.com/Papatheodorou-Group/GeneSpectra.git
cd GeneSpectra
```

[Pixi](https://pixi.sh/latest/) is used for dependency management.

First install [pixi](https://pixi.sh/latest/#installation). Then, run this command in the `GeneSpectra/` directory to install project dependencies:

```shell
pixi install -a
```

Note that the core gene classification code in GeneSpectra technically works on very basic Python and can be adapted to other environments. 

Installation should take about 5-10 minutes, mostly for conda to download packages. 

## Modules

### Metacells

Wrapper functions and helper functions to use [metacells](https://github.com/tanaylab/metacells) to create metacells based on scRNA-seq data. It is also recommended to follow the official metacells workflow to create the most tailored metacells anndata object (use the [iterative vignette](https://tanaylab.github.io/metacells-vignettes/iterative.html) for brand-new data), as you have more freedom to adjust various parameters. Alternatively, when the dataset is unsuitable for metacell calculation, merge cells of the same annotation label to create cell pools.

### Gene classification

Core module to perform gene filtering, normalisation, and gene specificity and distribution classification. Uses multi-processing to parallelise the processing of genes. Plotting functions of the gene class conservation heatmap is also included. 

### Cross-species

Cross-species comparison of gene classes and plotting. Using [ensembl](https://www.ensembl.org/index.html) or [eggNOG](http://eggnog5.embl.de/) homology.

## Running

### Example

A comprehensive running example of performing gene classification is provided at `run_classification_sum_cell_pools.py`

```shell
python run_classification_sum_cell_pools.py
```

### Expected output

A large table containing the specificity and distribution classes, and the GO annotations, of all genes in the anndata object. Cross-species orthology-mapped results and figures are also available if performed. 

### Expected run time

Depending on the dataset size, and if parallelisation is used, the running time is estimated to be between 10 and 60 minutes. 

### Data associated with preprint

The gene classification results for the three species datasets analysed in the preprint are publicly available at [Zenodo](https://doi.org/10.5281/zenodo.17077680).

### Reproducibility

Scripts and notebooks to recreate the analysis in the paper are available at [GeneSpectra_reproducibility](https://github.com/Papatheodorou-Group/GeneSpectra_reproducibility).

Developer/maintainer: Yuyao Song, <ysong@ebi.ac.uk>
