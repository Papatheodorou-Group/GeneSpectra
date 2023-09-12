# need to check number of cells in each cluster of the training set.
# /usr/bin/env python3
import SCCAF as sccaf
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import click


def get_cell_type_auc(clf, y_test, y_prob):
    rc_aucs = []  # AUC
    rp_aucs = []  # AUC from recall precision
    fprs = []  # FPR
    tprs = []  # TPR
    prss = []  # Precision
    recs = []  # Recall
    for i, cell_type in enumerate(clf.classes_):
        fpr, tpr, _ = sccaf.metrics.roc_curve(y_test == cell_type, y_prob[:, i])
        prs, rec, _ = sccaf.metrics.precision_recall_curve(y_test == cell_type, y_prob[:, i])
        fprs.append(fpr)
        tprs.append(tpr)
        prss.append(prs)
        recs.append(rec)
        rc_aucs.append(sccaf.metrics.auc(fpr, tpr))
        rp_aucs.append(sccaf.metrics.auc(rec, prs))
    tbl = pd.DataFrame(data=list(zip(clf.classes_, rp_aucs, rc_aucs)),
                       columns=['cell_group', "ROC_AUC", "PR_AUC"])
    return tbl


def get_clf_marker_vector(input_h5ad, species, cluster_key, topn=300, **kwargs):
    click.echo("Read input h5ad")
    input_ad = sc.read_h5ad(input_h5ad)

    click.echo("Preprocess to normalise and scale expression data")
    sc.pp.normalize_total(input_ad, target_sum=1e4)
    sc.pp.log1p(input_ad)
    sc.pp.highly_variable_genes(input_ad, min_mean=0.0125, max_mean=3, min_disp=0.5, **kwargs)
    sc.pp.scale(input_ad,
                max_value=10)
    # scale every gene to uni variance so that weights represent relative expression
    sc.pp.neighbors(input_ad, n_neighbors=10, n_pcs=50)
    sc.tl.umap(input_ad, min_dist=0.3)

    click.echo("Multinomial logistic model training")
    input_matrix = input_ad.X
    y_prob, y_pred, y_test, clf, cvsm, acc = sccaf.SCCAF_assessment(input_matrix, input_ad.obs[cluster_key], n=200)

    click.echo("Get AUCs per cluster")

    tbl1 = get_cell_type_auc(clf, y_test, y_prob)

    tbl1['test_acc'] = acc
    tbl1['CV_acc'] = cvsm
    tbl1['species'] = species
    tbl1['input_file'] = input_h5ad
    tbl1['cluster_key'] = cluster_key

    click.echo("Get markers vector per cluster")

    tbl2 = sccaf.get_topmarkers(clf, input_ad.var_names, topn=topn)
    click.echo("finish")

    return tbl1, tbl2
