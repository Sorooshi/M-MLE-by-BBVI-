import os
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
import mle_cls as mle
import ADmetrics as adm
import brute_ope as bope
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import datasets as skds
import cls_pw_ge as cls_pw_re
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
import tensorflow_probability as tfp
import synthetic_data_generator as sgd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import IsolationForest
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import marginalized_maximum_likelihood_estimation_by_bbvi as MMLE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import make_moons, make_circles, make_classification


tf.keras.backend.set_floatx('float64')
np.set_printoptions(suppress=True, precision=3)

tfk = tf.keras
tfpl = tfp.layers
tfkl = tf.keras.layers
tfd = tfp.distributions


def args_parser(args):
    run = args.run
    path = args.path
    n_units = args.n_units
    weights = args.weights
    if weights:
        weights = weights.split(", ")
        weights = [float(w) for w in weights]
    setting = args.setting
    n_epochs = args.n_epochs
    n_metrics = args.n_metrics
    n_samples = args.n_samples
    n_repeats = args.n_repeats
    n_classes = args.n_classes
    n_features = args.n_features
    batch_size = args.batch_size
    activation = args.activation
    name_of_exp = args.name_of_exp
    learning_rate = args.learning_rate
    dense_layer_type = args.dense_layer_type
    if dense_layer_type:
        dense_layer_type = dense_layer_type.lower()
    cluster_intermix_probs = args.cluster_intermix_probs
    if cluster_intermix_probs:
        cluster_intermix_probs = cluster_intermix_probs.split(", ")
        cluster_intermix_probs = [prob for prob in cluster_intermix_probs]

    return run, path, n_units, weights, setting, n_epochs,\
           n_metrics, n_samples, n_repeats, n_classes, n_features,\
           batch_size, activation, name_of_exp, learning_rate,\
           dense_layer_type, cluster_intermix_probs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run', type=int, default=0,
                        help='Whether to run the program or to evaluate the saved results')
    parser.add_argument('--path', type=str, default=None,
                        help='Path to load the data sets. '
                             'Default is None, and in this case'
                             'synthetic datasets will be used')
    parser.add_argument('--n_units', type=int, default=2,
                        help='An integer denoting number of neurons (units) '
                             'in the first layer of the Neural networks')
    parser.add_argument('--weights', type=str, default=None,
                        help='A str determining the trade off between '
                             'different classes representation '
                             'once make_classification data generator will be used,'
                             'e.g, "0.5, 0.5" creates a balanced dataset containing two classes,'
                             'while ".9, .05, .05," will creates an imbalanced '
                             'datasets containing three classes.')
    parser.add_argument('--setting', type=str, default='all',
                        help='Future use')
    parser.add_argument('--n_epochs', type=int, default=1500,
                        help='An integer denoting the number of epochs')
    parser.add_argument('--n_metrics', type=int, default=6,
                        help='An integer denoting number of metrics used '
                             'to evaluate the performance of algorithms')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='An integer denoting number of samples '
                             'in the case of synthetic datasets')
    parser.add_argument('--n_repeats', type=int, default=10,
                        help='An integer denoting number of repeats')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='An integer denoting the number of classes')
    parser.add_argument('--n_features', type=int, default=2,
                        help='An integer denoting the number of features')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='An integer denoting the batch size')
    parser.add_argument('--activation', type=str, default=None,
                        help='String determining which activation function should '
                             'be used once and algorithm is implemented in TF2.'
                             'Default value is None which applies Linear activation')
    parser.add_argument('--name_of_exp', type=str, default='3Small-Bal',
                        help='String denoting name of the experiment '
                             'in the case of synthetic datasets')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='A floating number denoting '
                             'the learning rate in used in optimizer')
    parser.add_argument('--dense_layer_type', type=str, default=None,
                        help='String denoting the type of Monte Carlo Gradient Estimator'
                             'which will be used during optimization')
    parser.add_argument('--cluster_intermix_probs', type=str,
                        default='.9, .5, .1',
                        help='A list of strings specifying the '
                             'cluster intermix probabilities; the smaller the harder')

    args = parser.parse_args()
    run, path, n_units, weights, setting, n_epochs, \
    n_metrics, n_samples, n_repeats, n_classes, n_features,\
    batch_size, activation, name_of_exp, learning_rate, \
    dense_layer_type, cluster_intermix_probs = args_parser(args)

    print("args:", weights, run, path)

    prepared = False
    if name_of_exp.lower() == 'mnist' or name_of_exp.lower() == 'kdd_90'\
            or name_of_exp.lower() == 'zerobias' or name_of_exp.lower() == 'egamma' or \
            name_of_exp.lower() == 'smuon' or name_of_exp.lower() == 'jetht':

        prepared = True

    if path is None and run == 1:
        print("Run SK-learn synthetic")

        h = .02  # step size in the mesh

        if dense_layer_type is not None:
            names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                     " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                     " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                     "MMLE_"+dense_layer_type]
        else:
            names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                     " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                     " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                     "MMLE_pw-li", "MMLE_fo-li", "MMLE_pw-re", "MMLE_fo-re"]

        if n_features <= 2:
            datasets_names = ["linearly_separable", "moon_shape", "circle_shape"]
        else:
            datasets_names = ["linearly_separable", ]

        classifiers = [
            KNeighborsClassifier(3,),
            SVC(kernel="linear", C=0.025, probability=True),
            DecisionTreeClassifier(max_depth=5,),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1,),
            AdaBoostClassifier(),
            GaussianNB(),
            mle,
            bope,
            cls_pw_re,
            IsolationForest(n_estimators=100, max_samples=2000,
                            max_features=1, bootstrap=False,
                            contamination='auto', n_jobs=-2),
            OneClassSVM(nu=0.01, kernel="rbf", gamma='scale', shrinking=True),
            MMLE,
            MMLE,
            MMLE,
            MMLE,
        ]

        outputs = {}
        stats = {}
        for dataset in datasets_names:
            for name in names:
                stats[dataset + "-" + name] = np.zeros([n_repeats, n_metrics])
                outputs[dataset + "-" + name] = {}

        for repeat in range(n_repeats):

            start = time.time()

            n_remaining = n_features - n_classes
            n_redundant = int(np.ceil(n_remaining/2))
            n_repeated = int(np.floor(n_remaining/2))

            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_classes,
                                       n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes,
                                       n_clusters_per_class=1, weights=weights, flip_y=0.4, class_sep=1.0,
                                       hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

            linearly_separable = (X, y)

            if n_features <= 2:
                datasets = [linearly_separable,
                            make_moons(n_samples=n_samples, noise=0.5, random_state=None),
                            make_circles(n_samples=n_samples, noise=0.5, factor=0.5, random_state=None),
                            ]
            else:
                datasets = [linearly_separable, ]

            figure = plt.figure(figsize=(30, 10))
            i = 1
            # iterate over datasets
            for ds_cnt, ds in enumerate(datasets):
                print("dataset:", datasets_names[ds_cnt], "repeat:", repeat)
                # preprocess dataset, split into training and test part
                X, y = ds
                X = StandardScaler().fit_transform(X)
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=.4, random_state=43)

                print("shapes:", X_train.shape, X_test.shape)

                x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))

                # just plot the dataset first
                cm = plt.cm.RdBu
                cm_bright = ListedColormap(['#FF0000', '#0000FF'])
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

                if ds_cnt == 0 and n_features <= 2:
                    ax.set_title("Input data")
                # Plot the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                           edgecolors='k')
                # Plot the testing points
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                           edgecolors='k', marker='+')
                ax.text(xx.max() - .3, yy.min() + .2, 'ROC-A.',
                        size=9, horizontalalignment='right')

                ax.text(xx.min() + 1.6, yy.min() + .2, 'Fscr.', size=9,
                        horizontalalignment='right')

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                i += 1

                # iterate over classifiers
                for name, clf in zip(names, classifiers):
                    outputs[datasets_names[ds_cnt] + "-" + name][repeat] = {}
                    print("Algorithm:", name)

                    one_hot_encoder = OneHotEncoder(sparse=False)
                    if n_features <= 2:
                        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

                    if name == "CrossEntropy":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_mle = mle.apply_mle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                            activation=activation, x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_logits = model_mle(X_test, training=False)
                        py_x_probs = tf.nn.softmax(py_x_logits)
                        labels_pred = tf.argmax(py_x_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt]+"-"+name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_pw-li":
                        dense_layer_type = "pw-li"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_MMLE = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt]+"-"+name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mmle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_fo-li":
                        dense_layer_type = 'fo-li'
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=True)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt]+"-"+name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mmle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_pw-re":
                        dense_layer_type = "pw-re"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt]+"-"+name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mmle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_fo-re":
                        dense_layer_type = "fo-re"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')

                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt]+"-"+name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mmle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == " Brute-OPE ":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_bope = bope.apply_bope(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                            activation=activation, x_train=X_train, y_train=y_train, x_val=None,
                            y_val=None, x_test=None, y_test=None, verbose=False)

                        py_x_logits = model_bope(X_test)
                        py_x_probs = tf.nn.softplus(py_x_logits)
                        labels_pred = [1 if i >= 0.5 else 0 for i in py_x_probs]
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=y_test,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = model_bope(
                                np.c_[xx.ravel(), yy.ravel()])
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == " CLS_PW-RE ":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_cls_ge = cls_pw_re.apply_cls_ge(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type='pw-re',
                            activation=activation, x_train=X_train, y_train=y_train_1h,
                            x_val=None, y_val=None, x_test=None, y_test=None, verbose=False
                            )

                        py_x_z_logits = model_cls_ge(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')

                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_cls_ge(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "Isl. Forest" or name == "  OneC-SVM ":
                        y_train_ = np.asarray([1 if i == 1 else -1 for i in y_train])
                        y_test_ = np.asarray([i if i == 1 else -1 for i in y_test])
                        clf.fit(X_train, y_train_)
                        py_x_probs = clf.decision_function(X_test).reshape(-1, 1)
                        labels_pred = clf.predict(X_test)
                        score = metrics.accuracy_score(y_true=y_test_, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(np.asarray(y_test_).reshape(-1, 1))
                        gt_ms_onehot = gt_ms_onehot.astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test_,
                            alg_probs=py_x_probs, gt_ms_onehot=y_test_,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(
                            y_true=y_test_, y_pred=labels_pred, average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

                    else:
                        clf.fit(X_train, y_train)
                        py_x_probs = clf.predict_proba(X_test)
                        labels_pred = clf.predict(X_test)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(np.asarray(y_test).reshape(-1, 1))
                        gt_ms_onehot = gt_ms_onehot.astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt]+"-"+name_of_exp,
                            alg_name=name+"-"+str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(
                            y_true=y_test, y_pred=labels_pred, average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            if hasattr(clf, "decision_function"):
                                print("decision_function")
                                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                            else:
                                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                    # Save the out_cmp
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 0] = prec
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 1] = rec
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 2] = fscr
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 3] = roc_auc
                    outputs[datasets_names[ds_cnt] + "-" + name][repeat] = labels_pred

                    # Put the result into a color plot
                    if n_features <= 2 and tf.is_tensor(Z):
                        Z = Z.numpy()
                    if n_features <= 2:
                        Z = Z.reshape(xx.shape)
                        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

                        # Plot the training points
                        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                                   edgecolors='k')
                        # Plot the testing points
                        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                                   edgecolors='k', alpha=0.6, marker='+')

                        ax.set_xlim(xx.min(), xx.max())
                        ax.set_ylim(yy.min(), yy.max())
                        ax.set_xticks(())
                        ax.set_yticks(())
                        if ds_cnt == 0:
                            ax.set_title(name)

                        ax.text(xx.max() - .3, yy.min() + .95, 'ROC-A.',
                                size=9, horizontalalignment='right')
                        ax.text(xx.max() - .3, yy.min() + .2, ('%.2f ' % roc_auc).lstrip('0'),
                                size=9, horizontalalignment='right')

                        ax.text(xx.min() + 1.6, yy.min() + .95, 'Fscr.', size=9,
                                horizontalalignment='right')
                        ax.text(xx.min() + 1.6, yy.min() + .2, ('%.2f ' % fscr).lstrip('0'),
                                size=9, horizontalalignment='right')
                        i += 1

                    end = time.time()

                    print("Repeat number:", repeat, "execution time of :", end - start)

            plt.tight_layout()
            plt.savefig("../figs/" + name_of_exp + str(repeat) + ".png")
            plt.close()

        with open(os.path.join('../data', "outputs_" + name_of_exp + ".pickle"), 'wb') as fp:
            pickle.dump(outputs, fp)

        with open(os.path.join('../data', "stats_" + name_of_exp + ".pickle"), 'wb') as fp:
            pickle.dump(stats, fp)

        for dataset in datasets_names:
            print("dataset:", dataset)
            print("\t \t", " FSCR(std) ", " RAUC.(std)",)  # " TNR.(std)", " PREC.(std)", " RCLL.(std)", " FSCR(std) ",
            for name in names:
                means = stats[dataset + "-" + name].mean(axis=0)
                stds = stats[dataset + "-" + name].std(axis=0)
                print(name, ": \t",
                      # "%.3f" % means[0], "%.3f" % stds[0],
                      # "%.3f" % means[1], "%.3f" % stds[1],
                      "%.3f" % means[2], "%.3f" % stds[2],
                      "%.3f" % means[3], "%.3f" % stds[3],
                      # "%.3f" % means[3], "%.3f" % stds[4],
                      # "%.3f" % means[3], "%.3f" % stds[5]
                      )

    elif path is None and run == 0:
        print("Evaluate synthetic")

        if dense_layer_type is not None:
            names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                     " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                     " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                     "MMLE_"+dense_layer_type]
        else:
            names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                     " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                     " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                     "MMLE_pw-li", "MMLE_fo-li", "MMLE_pw-re", "MMLE_fo-re"]

        if n_features <= 2:
            datasets_names = ["linearly_separable", "moon_shape", "circle_shape", ]
        else:
            datasets_names = ["linearly_separable", ]

        with open(os.path.join('../data', "outputs_" + name_of_exp
                                          + ".pickle"), 'rb') as fp:
            outputs = pickle.load(fp)

        with open(os.path.join('../data', "stats_" + name_of_exp
                                          + ".pickle"), 'rb') as fp:
            stats = pickle.load(fp)

        for dataset in datasets_names:
            print("dataset:", dataset)
            print("\t \t", " FSCR(std) ", " RAUC.(std)", )  # " FNR.(std)", " TNR.(std)", " PREC.(std)", " RCLL.(std)",
            for name in names:
                means = stats[dataset + "-" + name].mean(axis=0)
                stds = stats[dataset + "-" + name].std(axis=0)
                print(name, ": \t",
                      # "%.3f" % means[0], "%.3f" % stds[0],
                      # "%.3f" % means[1], "%.3f" % stds[1],
                      "%.3f" % means[2], "%.3f" % stds[2],
                      "%.3f" % means[3], "%.3f" % stds[3],
                      # "%.3f" % means[3], "%.3f" % stds[4],
                      # "%.3f" % means[3], "%.3f" % stds[5]
                      )

    elif path == "MVG_dist" and run == 1:

        print("Run cluster intermix investigation with Multivariate Gaussian Dist.")

        h = .02  # step size in the mesh

        if dense_layer_type is not None:
            names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                     " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                     " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                     "MMLE_" + dense_layer_type]
        else:
            names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                     " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                     " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                     "MMLE_pw-li", "MMLE_fo-li", "MMLE_pw-re", "MMLE_fo-re"]

        # a list of strings specifying cluster intermix
        # probabilities; the smaller the harder.
        datasets_names = cluster_intermix_probs

        classifiers = [
            KNeighborsClassifier(3, ),
            SVC(kernel="linear", C=0.025, probability=True),
            DecisionTreeClassifier(max_depth=5, ),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, ),
            AdaBoostClassifier(),
            GaussianNB(),
            mle,
            bope,
            cls_pw_re,
            IsolationForest(n_estimators=100, max_samples=2000,
                            max_features=1, bootstrap=False,
                            contamination='auto', n_jobs=-2),
            OneClassSVM(nu=0.01, kernel="rbf", gamma='scale', shrinking=True),
            MMLE,
            MMLE,
            MMLE,
            MMLE,
        ]

        outputs = {}
        stats = {}
        for dataset in datasets_names:
            for name in names:
                stats[dataset + "-" + name] = np.zeros([n_repeats, n_metrics])
                outputs[dataset + "-" + name] = {}

        for repeat in range(n_repeats):

            figure = plt.figure(figsize=(30, 10))
            i = 1
            # iterate over datasets
            for ds_cnt, ds in enumerate(datasets_names):
                print("dataset's intermix prob.:", datasets_names[ds_cnt], "repeat:", repeat)
                # Multivariate Gaussian Distribution with fewer parameters and better control
                # Quantitative features (only)
                cardinality = sgd.clusters_cardinality(N=n_samples, K=n_classes)
                X, Xn = sgd.generate_Y(N=n_samples, V=n_features, K=n_classes,
                                       pr_v=float(ds), cardinality=cardinality,
                                       features_type='Q', V_noise1=int(np.ceil(n_features/2))
                                       )
                y, _ = sgd.flat_ground_truth(cardinality)
                y = np.asarray(y)
                mvg_dist = (X, y)
                # preprocess dataset, split into training and test part
                X = StandardScaler().fit_transform(Xn)
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=.4, random_state=43)

                print("shapes:", X_train.shape, X_test.shape)

                x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
                y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                     np.arange(y_min, y_max, h))

                # just plot the dataset first
                cm = plt.cm.RdBu
                cm_bright = ListedColormap(['#FF0000', '#0000FF'])
                ax = plt.subplot(len(datasets_names), len(classifiers) + 1, i)

                if ds_cnt == 0 and n_features <= 2:
                    ax.set_title("Input data")
                # Plot the training points
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                           edgecolors='k')
                # Plot the testing points
                ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                           edgecolors='k', marker='+')
                ax.text(xx.max() - .3, yy.min() + .2, 'ROC-A.',
                        size=9, horizontalalignment='right')

                ax.text(xx.min() + 1.6, yy.min() + .2, 'Fscr.', size=9,
                        horizontalalignment='right')

                ax.set_xlim(xx.min(), xx.max())
                ax.set_ylim(yy.min(), yy.max())
                ax.set_xticks(())
                ax.set_yticks(())
                i += 1

                start = time.time()

                # iterate over classifiers
                for name, clf in zip(names, classifiers):
                    outputs[datasets_names[ds_cnt] + "-" + name][repeat] = {}
                    print("Algorithm:", name)

                    one_hot_encoder = OneHotEncoder(sparse=False)
                    if n_features <= 2:
                        ax = plt.subplot(len(datasets_names), len(classifiers) + 1, i)

                    if name == "CrossEntropy":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_mle = mle.apply_mle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                            activation=activation, x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_logits = model_mle(X_test, training=False)
                        py_x_probs = tf.nn.softmax(py_x_logits)
                        labels_pred = tf.argmax(py_x_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_pw-li":
                        dense_layer_type = "pw-li"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mmle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_fo-li":
                        dense_layer_type = 'fo-li'
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=True)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mmle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_pw-re":
                        dense_layer_type = "pw-re"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mmle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_fo-re":
                        dense_layer_type = "fo-re"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')

                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_mmle(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == " Brute-OPE ":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_bope = bope.apply_bope(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                            activation=activation, x_train=X_train, y_train=y_train, x_val=None,
                            y_val=None, x_test=None, y_test=None, verbose=False)

                        py_x_logits = model_bope(X_test)
                        py_x_probs = tf.nn.softplus(py_x_logits)
                        labels_pred = [1 if i >= 0.5 else 0 for i in py_x_probs]
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=y_test,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = model_bope(
                                np.c_[xx.ravel(), yy.ravel()])
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == " CLS_PW-RE ":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_cls_ge = cls_pw_re.apply_cls_ge(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type='pw-re',
                            activation=activation, x_train=X_train, y_train=y_train_1h,
                            x_val=None, y_val=None, x_test=None, y_test=None, verbose=False
                        )

                        py_x_z_logits = model_cls_ge(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')

                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = tf.nn.softmax(model_cls_ge(np.c_[xx.ravel(), yy.ravel()], training=False))[:, 1]
                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "Isl. Forest" or name == "  OneC-SVM ":
                        y_train_ = np.asarray([1 if i == 1 else -1 for i in y_train])
                        y_test_ = np.asarray([i if i == 1 else -1 for i in y_test])
                        clf.fit(X_train, y_train_)
                        py_x_probs = clf.decision_function(X_test).reshape(-1, 1)
                        labels_pred = clf.predict(X_test)
                        score = metrics.accuracy_score(y_true=y_test_, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(np.asarray(y_test_).reshape(-1, 1))
                        gt_ms_onehot = gt_ms_onehot.astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test_,
                            alg_probs=py_x_probs, gt_ms_onehot=y_test_,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(
                            y_true=y_test_, y_pred=labels_pred, average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

                    else:
                        clf.fit(X_train, y_train)
                        py_x_probs = clf.predict_proba(X_test)
                        labels_pred = clf.predict(X_test)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(np.asarray(y_test).reshape(-1, 1))
                        gt_ms_onehot = gt_ms_onehot.astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(
                            y_true=y_test, y_pred=labels_pred, average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        if n_features <= 2:
                            # Plot the decision boundary. For that, we will assign a color to each
                            # point in the mesh [x_min, x_max]x[y_min, y_max].
                            if hasattr(clf, "decision_function"):
                                print("decision_function")
                                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
                            else:
                                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

                    # Save the out_cmp
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 0] = prec
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 1] = rec
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 2] = fscr
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 3] = roc_auc
                    outputs[datasets_names[ds_cnt] + "-" + name][repeat] = labels_pred

                    # Put the result into a color plot
                    if n_features <= 2 and tf.is_tensor(Z):
                        Z = Z.numpy()
                    if n_features <= 2:
                        Z = Z.reshape(xx.shape)
                        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

                        # Plot the training points
                        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                                   edgecolors='k')
                        # Plot the testing points
                        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                                   edgecolors='k', alpha=0.6, marker='+')

                        ax.set_xlim(xx.min(), xx.max())
                        ax.set_ylim(yy.min(), yy.max())
                        ax.set_xticks(())
                        ax.set_yticks(())
                        if ds_cnt == 0:
                            ax.set_title(name)

                        ax.text(xx.max() - .3, yy.min() + .95, 'ROC-A.',
                                size=9, horizontalalignment='right')
                        ax.text(xx.max() - .3, yy.min() + .2, ('%.2f ' % roc_auc).lstrip('0'),
                                size=9, horizontalalignment='right')

                        ax.text(xx.min() + 1.6, yy.min() + .95, 'Fscr.', size=9,
                                horizontalalignment='right')
                        ax.text(xx.min() + 1.6, yy.min() + .2, ('%.2f ' % fscr).lstrip('0'),
                                size=9, horizontalalignment='right')
                        i += 1

                    end = time.time()
                    print("Repeat number:", repeat, "execution time of :", end - start)

            plt.tight_layout()
            plt.savefig("../figs/" + name_of_exp + str(repeat) + ".png")
            plt.close()

        with open(os.path.join('../data', "outputs_mvg" + name_of_exp + ".pickle"), 'wb') as fp:
            pickle.dump(outputs, fp)

        with open(os.path.join('../data', "stats_mvg" + name_of_exp + ".pickle"), 'wb') as fp:
            pickle.dump(stats, fp)

        for dataset in datasets_names:
            print("dataset:", dataset)
            print("\t \t", " FSCR(std) ", " RAUC.(std)", )  # " TNR.(std)", " PREC.(std)", " RCLL.(std)", " FSCR(std) ",
            for name in names:
                means = stats[dataset + "-" + name].mean(axis=0)
                stds = stats[dataset + "-" + name].std(axis=0)
                print(name, ": \t",
                      # "%.3f" % means[0], "%.3f" % stds[0],
                      # "%.3f" % means[1], "%.3f" % stds[1],
                      "%.3f" % means[2], "%.3f" % stds[2],
                      "%.3f" % means[3], "%.3f" % stds[3],
                      # "%.3f" % means[3], "%.3f" % stds[4],
                      # "%.3f" % means[3], "%.3f" % stds[5]
                      )

    elif path == "MVG_dist" and run == 0:

        print("Evaluate cluster intermix investigation with Multivariate Gaussian Dist.")

        if dense_layer_type is not None:
            names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                     " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                     " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                     "MMLE_"+dense_layer_type]
        else:
            names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                     " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                     " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                     "MMLE_pw-li", "MMLE_fo-li", "MMLE_pw-re", "MMLE_fo-re"]

        datasets_names = cluster_intermix_probs

        with open(os.path.join('../data', "outputs_mvg" + name_of_exp
                                          + ".pickle"), 'rb') as fp:
            outputs = pickle.load(fp)

        with open(os.path.join('../data', "stats_mvg" + name_of_exp
                                          + ".pickle"), 'rb') as fp:
            stats = pickle.load(fp)

        for dataset in datasets_names:
            print("dataset:", dataset)
            print("\t \t", " FSCR(std) ", " RAUC.(std)", )  # " FNR.(std)", " TNR.(std)", " PREC.(std)", " RCLL.(std)",
            for name in names:
                means = stats[dataset + "-" + name].mean(axis=0)
                stds = stats[dataset + "-" + name].std(axis=0)
                print(name, ": \t",
                      # "%.3f" % means[0], "%.3f" % stds[0],
                      # "%.3f" % means[1], "%.3f" % stds[1],
                      "%.3f" % means[2], "%.3f" % stds[2],
                      "%.3f" % means[3], "%.3f" % stds[3],
                      # "%.3f" % means[3], "%.3f" % stds[4],
                      # "%.3f" % means[3], "%.3f" % stds[5]
                      )

    elif path != "MVG_dist" and run == 1:  # path is not None and

        print("Run Real-World Dataset!")

        if prepared:
            with open(os.path.join(path, name_of_exp + ".pickle"), 'rb') as fp:
                DATA = pickle.load(fp)
            print("DATA is loaded", DATA.keys())

        elif name_of_exp.lower() == 'iris':
            print("Dataset under consideration:", name_of_exp)
            iris = skds.load_iris()
            X = iris.data.astype('float64')
            y = iris.target.astype('int64')
            n_classes = len(set(y))

        elif name_of_exp.lower() == 'wine':
            print("Dataset under consideration:", name_of_exp)
            wine = skds.load_wine()
            X = wine.data.astype('float64')
            y = wine.target.astype('int64')
            n_classes = len(set(y))

        elif name_of_exp.lower() == 'gcr':
            print("Dataset under consideration:", name_of_exp)
            DATA = np.loadtxt('/home/soroosh/FADA/data/gcr.npy')
            X = DATA[:, :-1].astype('float64')
            y = DATA[:, -1].astype('int64')
            n_classes = len(set(y))

        elif name_of_exp.lower() == 'mnist_org':
            print("Dataset under consideration:", name_of_exp)
            DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
            path = tf.keras.utils.get_file('mnist.npz', DATA_URL)
            with np.load(path) as data:
                samples_train = data['x_train'].astype('float64')
                X_train = np.zeros([60000, 784])
                X_test = np.zeros([10000, 784])
                y_train = data['y_train'].astype('int64')
                for x in range(6000):
                    X_train[x, :] = samples_train[x, :, :].flatten()

                samples_test = data['x_test'].astype('float64')
                for t in range(4000):
                    X_test[t, :] = samples_test[t, :, :].flatten()

                y_test = data['y_test'].astype('int64')
                print("MNIST:", X_train.shape, y_train.shape)
                n_classes = len(set(y_train))

        elif name_of_exp.lower() == 'bcw':  # Breast Cancer Wisconsin
            print("Dataset under consideration:", name_of_exp)
            bcw = skds.load_breast_cancer()
            X = bcw.data.astype('float64')
            y = bcw.target.astype('int64')
            n_classes = len(set(y))

        elif name_of_exp.lower() == 'covtype':
            print("Dataset under consideration:", name_of_exp)
            covtype = skds.fetch_covtype(data_home=path, )
            X = covtype.data.astype('float64')
            y = covtype.target.astype('int64')
            n_classes = len(set(y))

        # a list of strings specifying dataset names
        datasets_names = [name_of_exp]

        print("datasets:", datasets_names)

        if n_classes <= 2:
            if dense_layer_type is not None:
                names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                         " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                         " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                         "MMLE_" + dense_layer_type]
            else:
                names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                         " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                         " Brute-OPE ", " CLS_PW-RE ", "Isl. Forest", "  OneC-SVM ",
                         "MMLE_pw-li", "MMLE_fo-li", "MMLE_pw-re", "MMLE_fo-re"]

            classifiers = [
                KNeighborsClassifier(3, ),
                SVC(kernel="linear", C=0.025, probability=True),
                DecisionTreeClassifier(max_depth=5, ),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, ),
                AdaBoostClassifier(),
                GaussianNB(),
                mle,
                bope,
                cls_pw_re,
                IsolationForest(n_estimators=100, max_samples=2000,
                            max_features=1, bootstrap=False,
                            contamination='auto', n_jobs=-2),
                OneClassSVM(nu=0.01, kernel="rbf", gamma='scale', shrinking=True),
                MMLE,
                MMLE,
                MMLE,
                MMLE,
            ]

        else:
            if dense_layer_type is not None:
                names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                         " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                         " CLS_PW-RE ", "MMLE_" + dense_layer_type]
            else:
                names = ["N Neighbors", "Linear SVM ", "Dicsn. Tree", "Rnd. Forest",
                         " AdaBoost  ", "Naive Bayes", "CrossEntropy",
                         " CLS_PW-RE ",
                         "MMLE_pw-li", "MMLE_fo-li", "MMLE_pw-re", "MMLE_fo-re"]
            classifiers = [
                KNeighborsClassifier(3, ),
                SVC(kernel="linear", C=0.025, probability=True),
                DecisionTreeClassifier(max_depth=5, ),
                RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, ),
                AdaBoostClassifier(),
                GaussianNB(),
                mle,
                cls_pw_re,
                MMLE,
                MMLE,
                MMLE,
                MMLE,
            ]

        outputs = {}
        stats = {}
        # I have already split the real-world datasets into .98, .02 train, validation and, test splits
        setting = (0.98, 0.02)

        for dataset in datasets_names:
            for name in names:
                stats[dataset + "-" + name] = np.zeros([n_repeats, n_metrics])
                outputs[dataset + "-" + name] = {}

        for repeat in range(n_repeats):

            # iterate over datasets
            for ds_cnt, ds in enumerate(datasets_names):
                print("dataset's Name:", datasets_names[ds_cnt], "repeat:", repeat)
                if prepared:
                    X_train = DATA[setting][repeat]['X_tr'].astype('float64')
                    X_vlid = DATA[setting][repeat]['X_vl'].astype('float64')
                    X_test = DATA[setting][repeat]['X_ts'].astype('float64')

                    y_train = DATA[setting][repeat]['y_tr'].astype('float64')
                    y_vlid = DATA[setting][repeat]['y_vl'].astype('float64')
                    y_test = DATA[setting][repeat]['y_ts'].astype('float64')

                    # preprocess dataset, split into training and test part
                    X_train = StandardScaler().fit_transform(X_train)
                    X_test = StandardScaler().fit_transform(X_test)

                elif prepared is False and name_of_exp != 'mnist_org':
                    X = StandardScaler().fit_transform(X)
                    X_train, X_test, y_train, y_test = \
                        train_test_split(X, y, test_size=.4,)

                print("shapes:", X_train.shape, X_test.shape)
                n_features = X_train.shape[1]

                start = time.time()

                # iterate over classifiers
                for name, clf in zip(names, classifiers):
                    outputs[datasets_names[ds_cnt] + "-" + name][repeat] = {}
                    print("Algorithm:", name)

                    one_hot_encoder = OneHotEncoder(sparse=False)

                    if name == "CrossEntropy":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_mle = mle.apply_mle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                            activation=activation, x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_logits = model_mle(X_test, training=False)
                        py_x_probs = tf.nn.softmax(py_x_logits)
                        labels_pred = tf.argmax(py_x_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_pw-li":
                        dense_layer_type = "pw-li"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_fo-li":
                        dense_layer_type = 'fo-li'
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=True)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_pw-re":
                        dense_layer_type = "pw-re"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "MMLE_fo-re":
                        dense_layer_type = "fo-re"
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')

                        model_mmle = MMLE.apply_mmle(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type=dense_layer_type,
                            x_train=X_train, y_train=y_train_1h, x_val=None, y_val=None,
                            x_test=None, y_test=None, verbose=False)

                        py_x_z_logits = model_mmle(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')

                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == " Brute-OPE ":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_bope = bope.apply_bope(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size, learning_rate=learning_rate,
                            activation=activation, x_train=X_train, y_train=y_train, x_val=None,
                            y_val=None, x_test=None, y_test=None, verbose=False)

                        py_x_logits = model_bope(X_test)
                        py_x_probs = tf.nn.softplus(py_x_logits)
                        labels_pred = [1 if i >= 0.5 else 0 for i in py_x_probs]
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=y_test,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == " CLS_PW-RE ":
                        y_train_1h = one_hot_encoder.fit_transform(
                            y_train.reshape(-1, 1)).astype('float64')
                        model_cls_ge = cls_pw_re.apply_cls_ge(
                            n_units=n_units, n_features=n_features, n_classes=n_classes,
                            n_epochs=n_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, dense_layer_type='pw-re',
                            activation=activation, x_train=X_train, y_train=y_train_1h,
                            x_val=None, y_val=None, x_test=None, y_test=None, verbose=False
                        )

                        py_x_z_logits = model_cls_ge(X_test, training=False)
                        py_x_z_probs = tf.nn.softmax(py_x_z_logits)
                        labels_pred = tf.argmax(py_x_z_probs, axis=1)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(
                            np.asarray(y_test).reshape(-1, 1)).astype('float64')

                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_z_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_z_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(y_true=y_test, y_pred=labels_pred,
                                                                      average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                        y_train = one_hot_encoder.inverse_transform(y_train_1h).reshape(-1, 1)

                    elif name == "Isl. Forest" or name == "  OneC-SVM ":
                        y_train_ = np.asarray([1 if i == 1 else -1 for i in y_train])
                        y_test_ = np.asarray([i if i == 1 else -1 for i in y_test])
                        clf.fit(X_train, y_train_)
                        py_x_probs = clf.decision_function(X_test).reshape(-1, 1)
                        labels_pred = clf.predict(X_test)
                        score = metrics.accuracy_score(y_true=y_test_, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(np.asarray(y_test_).reshape(-1, 1))
                        gt_ms_onehot = gt_ms_onehot.astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test_,
                            alg_probs=py_x_probs, gt_ms_onehot=y_test_,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(
                            y_true=y_test_, y_pred=labels_pred, average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                    else:
                        clf.fit(X_train, y_train)
                        py_x_probs = clf.predict_proba(X_test)
                        labels_pred = clf.predict(X_test)
                        score = metrics.accuracy_score(y_true=y_test, y_pred=labels_pred)
                        gt_ms_onehot = one_hot_encoder.fit_transform(np.asarray(y_test).reshape(-1, 1))
                        gt_ms_onehot = gt_ms_onehot.astype('float64')
                        roc_auc = metrics.roc_auc_score(y_true=gt_ms_onehot, y_score=py_x_probs)
                        adm.plot_roc_auv_curve_of_an_algorithm(
                            alg_ms=labels_pred, gt_ms=y_test,
                            alg_probs=py_x_probs, gt_ms_onehot=gt_ms_onehot,
                            data_name=datasets_names[ds_cnt] + "-" + name_of_exp,
                            alg_name=name + "-" + str(repeat),
                            name_of_auc_roc_fig=name, sample_weight=None, case=0)

                        prf = metrics.precision_recall_fscore_support(
                            y_true=y_test, y_pred=labels_pred, average='weighted')
                        prec, rec, fscr = prf[0], prf[1], prf[2]

                    # Save the out_cmp
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 0] = prec
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 1] = rec
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 2] = fscr
                    stats[datasets_names[ds_cnt] + "-" + name][repeat, 3] = roc_auc
                    outputs[datasets_names[ds_cnt] + "-" + name][repeat] = labels_pred

                    end = time.time()
                    print("Repeat number:", repeat, "execution time of :", end - start)

        with open(os.path.join('../data', "outputs_real" + name_of_exp + ".pickle"), 'wb') as fp:
            pickle.dump(outputs, fp)

        with open(os.path.join('../data', "stats_real" + name_of_exp + ".pickle"), 'wb') as fp:
            pickle.dump(stats, fp)

        for dataset in datasets_names:
            print("dataset:", dataset)
            print("\t \t", " FSCR(std) ", " RAUC.(std)", )  # " TNR.(std)", " PREC.(std)", " RCLL.(std)", " FSCR(std) ",
            for name in names:
                means = stats[dataset + "-" + name].mean(axis=0)
                stds = stats[dataset + "-" + name].std(axis=0)
                print(name, ": \t",
                      # "%.3f" % means[0], "%.3f" % stds[0],
                      # "%.3f" % means[1], "%.3f" % stds[1],
                      "%.3f" % means[2], "%.3f" % stds[2],
                      "%.3f" % means[3], "%.3f" % stds[3],
                      # "%.3f" % means[3], "%.3f" % stds[4],
                      # "%.3f" % means[3], "%.3f" % stds[5]
                      )

    elif path != "MVG_dist" and run == 0:  # path is not None and

        print("Evaluate Real-World Dataset!")
