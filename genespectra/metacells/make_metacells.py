import anndata as ad
import matplotlib.pyplot as plt
import metacells as mc
import numpy as np
import os
import pandas as pd
import scipy.sparse as sp
import seaborn as sb
import scanpy as sc

from math import hypot
from matplotlib.collections import LineCollection

sb.set_style("white")


def exclude_mt_genes(input_ad, col_use):
    if col_use is not None and ~col_use in input_ad.var.columns:
        raise KeyError(
            f"{col_use} is not in {input_ad}.var.columns"
        )
    if col_use is None:
        return input_ad[:, ~input_ad.var_names.str.startswith("MT-", "Mt-", "mt-")]
    else:
        return input_ad[:, ~input_ad.var[col_use].str.startswith("MT", "Mt-", "mt-")]


def make_metacells(input_ad, target_umi, ):
    """
    Using metacell2 to create metacells from raw counts of scRNA-seq data

    :return: an anndata object with each cell as a metacell
    """
    print("make_metacells")


def check_anndata():
    print("check_anndata")


def summarize_counts():
    print("summarize_counts")


def plot_counts_density():
    print("plot_counts_density")
