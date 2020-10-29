"""

This is an module containing several evaluation metrics for ML4DC  studies on CERN data set.
 These studies are devoted to the task of abnormality detection on CERN's experimental results.
In these studies, Fedor Ratnikov is the Research Supervisor and Soroosh Shalileh is the Research Assistant.

In order to evaluate the performance of an algorithm, it is a common practice to run that algorithm
several times and compute the average and the standard deviation of the metric under consideration.
 Moreover, the supervised and semi-supervised algorithms need a training set, a validation set and a test set.


- Prepared by: Soroosh Shalileh.

- Contact Info:
    In the case of having comments please feel free to send an email to sr.shalileh@gmail.com

"""

import os
import numpy as np
from copy import deepcopy
from sklearn import metrics
import matplotlib.pyplot as plt

CASES = ['original', 'reconstructed', 'latent']


def evaluation_with_clustering_metrics_several(kmeans_ms, agg_ms, gmm_ms, gt_ms, case):

    recm_k, recm_a, recm_g = {}, {}, {}  # Results Evaluation with Clustering Metrics

    case_ = CASES[case]

    print(" ")
    print("Results on the " + case_ + " Variable")

    print("\t", "  tr", "  ts", " ", "\t",
          "  K-ARI  ", "  K-NMI ", "\t",
          "  A-ARI  ", "  A-NMI ", "\t",
          "  G-ARI  ", "  G-NMI  ",
          )

    print(" \t", " \t", " \t",
          " Ave ", " std  ", " Ave ", "std ",
          " Ave ", " std  ", " Ave ", "std ",
          " Ave ", " std  ", " Ave ", "std"
          )

    for setting, results in kmeans_ms.items():

        ARI_k, NMI_k, ARI_a, NMI_a, ARI_g, NMI_g = [], [], [], [], [], []
        for repeat, result in results.items():
            gt = gt_ms[setting][repeat]
            lp_k = result[case]
            lp_a = agg_ms[setting][repeat][case]
            lp_g = gmm_ms[setting][repeat][case]
            ARI_k.append(metrics.adjusted_rand_score(gt, lp_k))
            NMI_k.append(metrics.normalized_mutual_info_score(gt, lp_k, average_method='max'))

            ARI_a.append(metrics.adjusted_rand_score(gt, lp_a))
            NMI_a.append(metrics.normalized_mutual_info_score(gt, lp_a, average_method='max'))

            ARI_g.append(metrics.adjusted_rand_score(gt, lp_g))
            NMI_g.append(metrics.normalized_mutual_info_score(gt, lp_g, average_method='max'))

        ari_ave_k = np.mean(np.asarray(ARI_k), axis=0)
        ari_std_k = np.std(np.asarray(ARI_k), axis=0)
        nmi_ave_k = np.mean(np.asarray(NMI_k), axis=0)
        nmi_std_k = np.std(np.asarray(NMI_k), axis=0)
        recm_k[setting] = [ari_ave_k, ari_std_k, nmi_ave_k, nmi_std_k]  # Evaluation Results Clustering Kmeans
        # recm_k = [ari_ave_k, ari_std_k, nmi_ave_k, nmi_std_k]  # Evaluation Results Clustering Kmeans

        ari_ave_a = np.mean(np.asarray(ARI_a), axis=0)
        ari_std_a = np.std(np.asarray(ARI_a), axis=0)
        nmi_ave_a = np.mean(np.asarray(NMI_a), axis=0)
        nmi_std_a = np.std(np.asarray(NMI_a), axis=0)
        recm_a[setting] = [ari_ave_a, ari_std_a, nmi_ave_a, nmi_std_a]  # Evaluation Results Clustering Agglomerative
        # recm_a = [ari_ave_a, ari_std_a, nmi_ave_a, nmi_std_a]  # Evaluation Results Clustering Agglomerative

        ari_ave_g = np.mean(np.asarray(ARI_g), axis=0)
        ari_std_g = np.std(np.asarray(ARI_g), axis=0)
        nmi_ave_g = np.mean(np.asarray(NMI_g), axis=0)
        nmi_std_g = np.std(np.asarray(NMI_g), axis=0)
        recm_g[setting] = [ari_ave_g, ari_std_g, nmi_ave_g, nmi_std_g]  # Evaluation Results Clustering GMM
        # recm_g = [ari_ave_g, ari_std_g, nmi_ave_g, nmi_std_g]  # Evaluation Results Clustering GMM

    return recm_k, recm_a, recm_g


def perf_measure(y_true, y_pred, weighted=True):

    cnf_matrix = metrics.confusion_matrix(y_true, y_pred)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    NUM_LPs = cnf_matrix.sum(axis=0)
    NUM_TPs = cnf_matrix.sum(axis=1)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    TNR_W = np.dot(TNR, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    TPR_W = np.dot(TPR, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    PPV_W = np.dot(PPV, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    # Negative predictive value
    NPV = TN / (TN + FN)
    NPV_W = np.dot(NPV, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    FPR_W = np.dot(FPR, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    # False negative rate
    FNR = FN / (TP + FN)
    FNR_W = np.dot(FNR, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    # False discovery rate
    FDR = FP / (TP + FP)
    FDR_W = np.dot(FDR, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    ACC_W = np.dot(ACC, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    FSCORE = np.divide((2 * PPV * TPR), (PPV + TPR))
    FSCORE_W = np.dot(FSCORE, NUM_TPs) / np.sum(NUM_TPs)  # Weighted

    # PPV, TPR, FSCORE, FNR, FPR, TNR
    # PPV_W, TPR_W, FSCORE_W, FNR_W, FPR_W, TNR_W
    if weighted is True:
        return FNR_W, TNR_W
    else:
        return FNR, TNR


def evaluation_with_classification_metric_several(kmeans_ms, agg_ms, gmm_ms, gt_ms, case):

    recm_k, recm_a, recm_g = {}, {}, {}  # Results Evaluation with Classification Metrics
    case_ = CASES[case]

    print(" ")
    print(" classification metrics ")
    print("Results on the " + case_ + " Variable")
    print(" ")
    print("\t", "  tr", "  ts", " ",
          "\tK-F-Score ", "    K-FNR ", "   K-TNR",
          "   A-F-Score ", "    A-FNR ", "   A-TNR ",
          "     G-F-Score  ", "  G-FNR ", "    G-TNR ",
          )
    print(" \t", " \t", "\t"
          "Ave ", " std ", " Ave ", " std ", "Ave ", " std ",
          "Ave ", " std ", " Ave ", "  std ", " Ave ", " std ",
          "Ave ", " std ", " Ave ", "  std ", " Ave ", " std"
          )

    for setting, results in kmeans_ms.items():

        precision_k, recall_k, fscore_k, roc_auc_k, fnr_k, tnr_k = [], [], [], [], [], []
        precision_a, recall_a, fscore_a, roc_auc_a, fnr_a, tnr_a = [], [], [], [], [], []
        precision_g, recall_g, fscore_g, roc_auc_g, fnr_g, tnr_g = [], [], [], [], [], []

        for repeat, result in results.items():

            gt = gt_ms[setting][repeat]
            lp_k = result[case]
            lp_a = agg_ms[setting][repeat][case]
            lp_g = gmm_ms[setting][repeat][case]

            tmp_k = metrics.precision_recall_fscore_support(gt, lp_k, average='weighted')
            tmp_k_ = perf_measure(gt, lp_k, weighted=True)
            roc_auc_k.append(metrics.roc_auc_score(gt, lp_k, average='weighted'))

            tmp_a = metrics.precision_recall_fscore_support(gt, lp_a, average='weighted')
            tmp_a_ = perf_measure(gt, lp_a, weighted=True)
            roc_auc_a.append(metrics.roc_auc_score(gt, lp_a, average='weighted'))

            tmp_g = metrics.precision_recall_fscore_support(gt, lp_g, average='weighted')
            tmp_g_ = perf_measure(gt, lp_g, weighted=True)
            roc_auc_g.append(metrics.roc_auc_score(gt, lp_g, average='weighted'))

            precision_k.append(tmp_k[0])
            recall_k.append(tmp_k[1])
            fscore_k.append(tmp_k[2])
            fnr_k.append(tmp_k_[0])
            tnr_k.append(tmp_k_[1])

            precision_a.append(tmp_a[0])
            recall_a.append(tmp_a[1])
            fscore_a.append(tmp_a[2])
            fnr_a.append(tmp_a_[0])
            tnr_a.append(tmp_a_[1])

            precision_g.append(tmp_g[0])
            recall_g.append(tmp_g[1])
            fscore_g.append(tmp_g[2])
            fnr_g.append(tmp_g_[0])
            tnr_g.append(tmp_g_[1])

        # K-means stats
        precision_ave_k = np.mean(np.asarray(precision_k), axis=0)
        precision_std_k = np.std(np.asarray(precision_k), axis=0)

        recall_ave_k = np.mean(np.asarray(recall_k), axis=0)
        recall_std_k = np.std(np.asarray(recall_k), axis=0)

        fscore_ave_k = np.mean(np.asarray(fscore_k), axis=0)
        fscore_std_k = np.std(np.asarray(fscore_k), axis=0)

        roc_auc_ave_k = np.mean(np.asarray(roc_auc_k), axis=0)
        roc_auc_std_k = np.std(np.asarray(roc_auc_k), axis=0)

        fnr_ave_k = np.mean(np.asarray(fnr_k), axis=0)
        fnr_std_k = np.std(np.asarray(tnr_k), axis=0)
        tnr_ave_k = np.mean(np.asarray(tnr_k), axis=0)
        tnr_std_k = np.std(np.asarray(tnr_k), axis=0)
        recm_k[setting] = [fscore_ave_k, fscore_std_k, fnr_ave_k, fnr_std_k, tnr_ave_k, tnr_std_k]

        # Agglomerative stats
        precision_ave_a = np.mean(np.asarray(precision_a), axis=0)
        precision_std_a = np.std(np.asarray(precision_a), axis=0)

        recall_ave_a = np.mean(np.asarray(recall_a), axis=0)
        recall_std_a = np.std(np.asarray(recall_a), axis=0)

        fscore_ave_a = np.mean(np.asarray(fscore_a), axis=0)
        fscore_std_a = np.std(np.asarray(fscore_a), axis=0)

        roc_auc_ave_a = np.mean(np.asarray(roc_auc_a), axis=0)
        roc_auc_std_a = np.std(np.asarray(roc_auc_a), axis=0)

        fnr_ave_a = np.mean(np.asarray(fnr_a), axis=0)
        fnr_std_a = np.std(np.asarray(tnr_a), axis=0)
        tnr_ave_a = np.mean(np.asarray(tnr_a), axis=0)
        tnr_std_a = np.std(np.asarray(tnr_a), axis=0)
        recm_a[setting] = [fscore_ave_a, fscore_std_a, fnr_ave_a, fnr_std_a, tnr_ave_a, tnr_std_a]

        # GMM stats
        precision_ave_g = np.mean(np.asarray(precision_g), axis=0)
        precision_std_g = np.std(np.asarray(precision_g), axis=0)

        recall_ave_g = np.mean(np.asarray(recall_g), axis=0)
        recall_std_g = np.std(np.asarray(recall_g), axis=0)

        fscore_ave_g = np.mean(np.asarray(fscore_g), axis=0)
        fscore_std_g = np.std(np.asarray(fscore_g), axis=0)

        roc_auc_ave_g = np.mean(np.asarray(roc_auc_g), axis=0)
        roc_auc_std_g = np.std(np.asarray(roc_auc_g), axis=0)

        fnr_ave_g = np.mean(np.asarray(fnr_g), axis=0)
        fnr_std_g = np.std(np.asarray(tnr_g), axis=0)
        tnr_ave_g = np.mean(np.asarray(tnr_g), axis=0)
        tnr_std_g = np.std(np.asarray(tnr_g), axis=0)
        recm_g[setting] = [fscore_ave_g, fscore_std_g, fnr_ave_g, fnr_std_g, tnr_ave_g, tnr_std_g]

    return recm_k, recm_a, recm_g


def evaluation_with_clustering_metrics(alg_ms, gt_ms, case):
    
    recm_k = {}  # Results Evaluation with Clustering Metrics

    case_ = CASES[case]

    print(" ")
    print("Results on the " + case_ + " Variable")

    print("\t", "  p", "  q", " a/e", "\t",
          "  K-ARI  ", "  K-NMI ", "\t",
          )

    print(" \t", " \t", " \t",
          " Ave ", " std  ", " Ave ", "std ",
          )

    for setting, results in alg_ms.items():

        ARI_k, NMI_k = [], []

        for repeat, result in results.items():
            gt = gt_ms[setting][repeat]
            lp_k = result
            ARI_k.append(metrics.adjusted_rand_score(gt, lp_k))
            NMI_k.append(metrics.normalized_mutual_info_score(gt, lp_k, average_method='max'))

        ari_ave_k = np.mean(np.asarray(ARI_k), axis=0)
        ari_std_k = np.std(np.asarray(ARI_k), axis=0)
        nmi_ave_k = np.mean(np.asarray(NMI_k), axis=0)
        nmi_std_k = np.std(np.asarray(NMI_k), axis=0)
        recm_k[setting] = [ari_ave_k, ari_std_k, nmi_ave_k, nmi_std_k]  # Evaluation Results Clustering Kmeans
        # recm_k = [ari_ave_k, ari_std_k, nmi_ave_k, nmi_std_k]  # Evaluation Results Clustering Kmeans

    return recm_k


def evaluation_with_classification_metric(alg_ms, gt_ms, case):

    recm_k = {}  # Results Evaluation with Classification Metrics
    case_ = CASES[case]

    print(" ")
    print("Results on the " + case_ + " Variable")

    print(" ")
    print("Results on the" + 'f' + "Variable")
    print("\t", "  p", "  q", " a/e   ",
          "\tK-F-Score ", "    K-FNR ", "   K-TNR",
          )
    print(" \t", " \t", " \t",  "Ave ", " std ", " Ave ", " std ", "Ave ", " std ",)

    for setting, results in alg_ms.items():

        precision_k, recall_k, fscore_k, roc_auc_k, fnr_k, tnr_k = [], [], [], [], [], []

        for repeat, result in results.items():
            gt = gt_ms[setting][repeat]
            lp_k = result

            tmp_k = metrics.precision_recall_fscore_support(gt, lp_k, average='weighted')
            roc_auc_k.append(metrics.roc_auc_score(gt, lp_k, average='weighted')
                             if metrics.roc_auc_score(gt, lp_k, average='weighted') > 0.5
                             else 1-metrics.roc_auc_score(gt, lp_k, average='weighted'))

            tmp_k_ = perf_measure(gt, lp_k, weighted=True)
            roc_auc_k.append(metrics.roc_auc_score(gt, lp_k, average='weighted'))

            precision_k.append(tmp_k[0])
            recall_k.append(tmp_k[1])
            fscore_k.append(tmp_k[2])
            fnr_k.append(tmp_k_[0])
            tnr_k.append(tmp_k_[1])

        # Algorithm's stats
        precision_ave_k = np.mean(np.asarray(precision_k), axis=0)
        precision_std_k = np.std(np.asarray(precision_k), axis=0)

        recall_ave_k = np.mean(np.asarray(recall_k), axis=0)
        recall_std_k = np.std(np.asarray(recall_k), axis=0)

        fscore_ave_k = np.mean(np.asarray(fscore_k), axis=0)
        fscore_std_k = np.std(np.asarray(fscore_k), axis=0)

        roc_auc_ave_k = np.mean(np.asarray(roc_auc_k), axis=0)
        roc_auc_std_k = np.std(np.asarray(roc_auc_k), axis=0)

        fnr_ave_k = np.mean(np.asarray(fnr_k), axis=0)
        fnr_std_k = np.std(np.asarray(tnr_k), axis=0)
        tnr_ave_k = np.mean(np.asarray(tnr_k), axis=0)
        tnr_std_k = np.std(np.asarray(tnr_k), axis=0)
        recm_k[setting] = [fscore_ave_k, fscore_std_k, fnr_ave_k, fnr_std_k, tnr_ave_k, tnr_std_k]

    return recm_k


def plot_roc_auv_curves_of_an_algorithm(alg_ms, gt_ms, data_name, alg_name, case, sample_weight=None):

    # fixed recall values
    rec_values = [0.8, 0.9, 0.95, 0.99]

    precisions = []
    recalls = []
    roc_auc_scores = []
    average_precision_scores = []

    case_ = CASES[case]

    print(" ")
    print("Results on the " + case_ + " Variable")

    for setting, results in alg_ms.items():
        for repeat, result in results.items():

            y_test = gt_ms[setting][repeat]  # ground truth

            if -1 in y_test:
                y_test = [0 if i == -1 else 1 for i in y_test]

            y_test = np.asarray(y_test)

            y_pred = deepcopy(result)  # predicted labels
            if -1 in y_pred:
                y_pred = [0 if i == -1 else 1 for i in y_pred]

            y_pred = np.asarray(y_pred)

            fig, axes = plt.subplots(nrows=3, figsize=(6, 15))
            ax = axes[0]
            ax.grid(True)
            precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred, )

            print("recalls values", rec_values)
            prec_values = []

            for v in rec_values:
                prec_values.append(max(precision[recall > v]))

            print("precision_values", prec_values)

            average_precision_score = metrics.average_precision_score(y_test, y_pred,
                                                                      sample_weight=sample_weight,
                                                                      average='weighted')

            print("average_precision_score", average_precision_score)

            roc_auc_score = metrics.roc_auc_score(y_test, y_pred, average='weighted')

            print("roc_auc_score", roc_auc_score)

            ax.step(recall, precision, color='b', alpha=0.2,
                    where='post')
            ax.fill_between(recall, precision, step='post', alpha=0.2,
                            color='b')
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)
            ax.set_title('Data set:' + str(repeat) + ' Precision-Recall curve: AP={0:0.3f}'.format(
                average_precision_score), fontsize=15)

            ax = axes[1]
            ax.grid(True)
            fpr, tpr, _ = metrics.roc_curve(y_test, y_pred, sample_weight=sample_weight)
            ax.plot(fpr, tpr)
            ax.set_title('ROC AUC curve: roc_auc ={0:0.3f}'.format(roc_auc_score), fontsize=15)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.set_xlabel('FPR', fontsize=10)
            ax.set_ylabel('TPR', fontsize=10)

            ax = axes[2]
            bad_test = len(y_test) - np.sum(y_test)  # bad labels are zeros
            good_test = np.sum(y_test)  # good labels are ones
            ax.plot(sorted(y_pred[np.where(y_test == 1.)[0]], reverse=True),
                    np.arange(good_test) / good_test * 100, label="good")

            ax.plot(sorted(y_pred[np.where(y_test == 0.)[0]]), np.arange(bad_test) / bad_test * 100, label="bad")
            ax.set_title('Predicted probability', fontsize=25)
            ax.tick_params(axis='x', labelsize=15)
            ax.tick_params(axis='y', labelsize=15)

            fig.subplots_adjust(hspace=0.5)
            plt.legend()
            plt.grid(True)

            # path = '/home/sshalileh/ml4dc/fig'
            path = 'figs'
            if not os.path.exists(path):
                os.mkdir(path)

            dir_path = os.path.join(path, data_name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)

            fig.savefig(os.path.join(dir_path, alg_name + '-' +
                                     str(repeat) + '-' + CASES[case] + '.png'))

            plt.close()

        precisions.append(prec_values)
        recalls.append(recall)
        roc_auc_scores.append(roc_auc_score)
        average_precision_scores.append(average_precision_score)

    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    roc_auc_scores = np.asarray(roc_auc_scores)
    average_precision_scores = np.asarray(average_precision_scores)

    return None


def plot_roc_auv_curve_of_an_algorithm(alg_ms, gt_ms, alg_probs, gt_ms_onehot, data_name, alg_name,
                                       name_of_auc_roc_fig, sample_weight=None, case=0):

    case_ = CASES[case]

    # ground truth
    if -1 in gt_ms:
        y_test = [0 if i == -1 else 1 for i in gt_ms]
    else:
        y_test = gt_ms

    y_test = np.asarray(y_test)

    # predicted labels
    if -1 in alg_ms:
        y_pred = [0 if i == -1 else 1 for i in alg_ms]
    else:
        y_pred = alg_ms

    y_pred = np.asarray(y_pred)
    n_classes = len(set(y_test))

    fig, axes = plt.subplots(nrows=3, figsize=(6, 15))
    ax = axes[0]
    ax.grid(True)

    if n_classes <= 2:

        precision, recall, fscore = metrics.precision_recall_curve(y_test, y_pred, )
        roc_auc_score = metrics.roc_auc_score(gt_ms_onehot, alg_probs, average='weighted')
        average_precision_score = metrics.average_precision_score(y_test, y_pred,
                                                                  sample_weight=sample_weight,
                                                                  average='weighted')
        ax.step(recall, precision, color='b', alpha=0.2,
                where='post')
        ax.fill_between(recall, precision, step='post', alpha=0.2,
                        color='b')
        ax.set_xlabel('Recall', fontsize=10)
        ax.set_ylabel('Precision', fontsize=10)
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.05])
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_title('Dataset: ' + data_name + ' Precision-Recall curve: AP={0:0.3f}'.format(
            average_precision_score), fontsize=11)

        ax = axes[1]
        ax.grid(True)
        fpr, tpr, thr = metrics.roc_curve(y_test, y_pred, sample_weight=sample_weight)
        ax.plot(fpr, tpr)
        ax.set_title('ROC AUC curve: roc_auc ={0:0.3f}'.format(roc_auc_score), fontsize=15)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel('FPR', fontsize=10)
        ax.set_ylabel('TPR', fontsize=10)

        ax = axes[2]
        bad_test = len(y_test) - np.sum(y_test)  # bad labels are zeros
        good_test = int(np.sum(y_test))  # good labels are ones
        ax.plot(sorted(y_pred[np.where(y_test == 1.)[0]], reverse=True),
                np.arange(good_test) / good_test * 100, label="good")

        ax.plot(sorted(y_pred[np.where(y_test == 0.)[0]]),
                np.arange(bad_test) / bad_test * 100, label="bad")
        ax.set_title('Predicted probability', fontsize=25)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        fig.subplots_adjust(hspace=0.5)
        plt.legend()
        plt.grid(True)

        # path = '/home/sshalileh/ml4dc/fig'
        path = '../figs'
        if not os.path.exists(path):
            os.mkdir(path)

        dir_path = os.path.join(path, data_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        fig.savefig(os.path.join(dir_path, alg_name + '.png'))  # name_of_auc_roc_fig >> alg_name

        plt.close()

    else:
        roc_auc_score = metrics.roc_auc_score(gt_ms_onehot, alg_probs, multi_class='ovo', average='macro')
        # average_precision_score = metrics.average_precision_score(y_test, y_pred,
        #                                                       sample_weight=sample_weight,
        #                                                       average='weighted')

        precision, recall, fscore, fpr, tpr = 0, 0, 0, 0, 0,

    return precision, recall, fscore, fpr, tpr, roc_auc_score


def plot_train_valid_curves_of_algorithm(acc, val_acc, v, C, loss, val_loss,
                                         n_epochs, data_name, name_of_train_val_fig):

    epochs_range = range(n_epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy ' + str(v) + '/' + str(len(C)))

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss ' + str(v) + '/' + str(len(C)))

    # path = '/home/sshalileh/ml4dc/fig'
    path = '../figs'
    if not os.path.exists(path):
        os.mkdir(path)

    dir_path = os.path.join(path, data_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    plt.savefig(os.path.join(dir_path, name_of_train_val_fig + '.png'))

