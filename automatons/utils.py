from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, mixture
from sklearn.neighbors import kneighbors_graph
from itertools import cycle, islice
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
import pandas as pd
import seaborn as sns
import sys
from scipy import stats
orig_stdout = sys.stdout

def plot_pca(X, y):

    cov = np.cov(X.T)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    tot = sum(eigen_values)
    var_exp = [(i / tot) for i in sorted(eigen_values, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    f, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.bar(range(X.shape[1]), var_exp, alpha=0.5, align='center', label='Individual variance explained')
    ax1.step(range(X.shape[1]), cum_var_exp, where='mid', label='Cumulative variance explained')
    ax1.legend(loc='best')
    ax1.axis('tight')
    ax1.set_xlabel('n_components')
    ax1.set_ylabel('explained variance ratio')

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    # Plot the training points
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
                edgecolor='k')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)


    X_reduced = PCA(n_components=2).fit_transform(X)
    ax3.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.Set1,
                edgecolor='k')
    ax3.axis('tight')
    ax3.set_title("First three PCA directions")
    ax3.set_xlabel('First PCA Component')
    ax3.set_ylabel('Second PCA Component')
    f.savefig('pca.png')







# ============
# Set up cluster parameters
# ============



def plot_clusters(X, y):
    # update parameters with dataset-specific values

    params = {'quantile': .3,
                'eps': .3,
                'damping': .9,
                'preference': -200,
                'n_neighbors': 20,
                'n_clusters': len(np.unique(y)),
                'n_components': len(np.unique(y)),
                'init':'pca'}

    plt.figure()
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                        hspace=.01)


    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(
        n_clusters=params['n_clusters'], linkage='ward',
        connectivity=connectivity)
    spectral = cluster.SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="nearest_neighbors")
    dbscan = cluster.DBSCAN(eps=params['eps'])
    affinity_propagation = cluster.AffinityPropagation(
        damping=params['damping'], preference=params['preference'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')
    #tsne = TSNE(n_components=params['n_components'], init='pca', random_state=0)

    clustering_algorithms = (
        ('MiniBatchKMeans', two_means),
        ('AffinityPropagation', affinity_propagation),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('Birch', birch),
        ('GaussianMixture', gmm),
    )
    index = 1
    for name, algorithm in clustering_algorithms:
        t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the " +
                "connectivity matrix is [0-9]{1,2}" +
                " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding" +
                " may not work as expected.",
                category=UserWarning)
            algorithm.fit(X)

        t1 = time.time()
        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(np.int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(1, len(clustering_algorithms), index)
        index += 1

        plt.title(name, size=18)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')

    plt.savefig('clustering.png')



def print_classfication_report(classifier, y_true, y_pred, stem='clf'):
    f = open(stem + '_report.txt', 'w')
    sys.stdout = f
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_true, y_pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_true, y_pred))
    sys.stdout = orig_stdout
    f.close()
    print("Classification report for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(y_true, y_pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_true, y_pred))



def one_hot_encoder(labels):
    labels = np.int32(labels)
    n_values = np.max(labels) + 1
    return np.eye(n_values)[labels]



def pr_curve(Y_test, y_score, n_classes, stem='train'):

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    yt = one_hot_encoder(Y_test)
    ys = one_hot_encoder(y_score)
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(yt[:, i],
                                                            ys[:, i])
        average_precision[i] = average_precision_score(yt[:, i], ys[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(yt.ravel(),
                                                                    ys.ravel())
    average_precision["micro"] = average_precision_score(yt, ys,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AUC={0:0.2f}'
            .format(average_precision["micro"]))
    plt.savefig(stem + '_precision_recall_curve.png')

def plot_feature_importance(X, y):
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances - Trees")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlabel('Feature Number')
    plt.ylabel('Mean Decrease in Accuracy (MDA)')
    plt.xlim([-1, X.shape[1]])
    plt.savefig('Trees_features_ranking.png')

    n_features = X.shape[1]
    f_test, _ = f_classif(X, y)
    f_test /= np.max(f_test)
    indices_ft = np.argsort(f_test)[::-1]

    mi = mutual_info_classif(X, y)
    mi /= np.max(mi)
    indices_mi = np.argsort(mi)[::-1]


    plt.figure()
    plt.title("Feature importances F-Score")
    plt.bar(range(n_features), f_test[indices_ft],
            color="r", align="center")
    plt.xticks(range(n_features), indices_ft)
    plt.xlabel('Feature Number')
    plt.ylabel('Importance')
    plt.xlim([-1, n_features])
    plt.savefig('F-Score-features_ranking.png')


    plt.figure()
    plt.title("Feature importances F-Score")
    plt.bar(range(n_features), mi[indices_mi],
            color="r", align="center")
    plt.xticks(range(n_features), indices_mi)
    plt.xlabel('Feature Number')
    plt.ylabel('Importance')
    plt.xlim([-1, n_features])
    plt.savefig('MI_features_ranking.png')

    importances /= np.max(importances)
    comb_imp = importances + mi + f_test
    comb_imp /= np.max(comb_imp)
    indices_ci = np.argsort(comb_imp)[::-1]

    plt.figure()
    plt.title("Feature importances Combined")
    plt.bar(range(n_features), comb_imp[indices_ci],
            color="r", align="center")
    plt.xticks(range(n_features), indices_ci)
    plt.xlabel('Feature Number')
    plt.ylabel('Importance')
    plt.xlim([-1, n_features])
    plt.savefig('combined_features_ranking.png')

    return indices_ci


def plot_feature_corr(X, stem='train'):
    df = pd.DataFrame(X)
    plt.figure()
    sns_plot = sns.pairplot(df)
    sns_plot.savefig(stem + '_pair_plot.png')
    f2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), ax=ax2, vmin=-1, vmax=1, cmap=plt.cm.RdBu, annot=True)
    f2.savefig(stem + '_corr_plot.png')
    f3, ax3 = plt.subplots()
    sns.violinplot(data=df, ax=ax3)
    f3.savefig(stem + '_voilin_plot.png')


def read_blocks(text_fname):
    f = open(text_fname, 'r')
    lines = f.readlines()
    f.close()

    blocks = []
    rows = []
    for line in lines:
        line = line.strip()
        if  bool(line):
            rows.append(np.array([float(d) for d in line.split()]))
        else:
            blocks.append(np.vstack(rows))
            rows = []
    blocks = np.array(blocks)
    return blocks

def read_data():
    blocks = read_blocks('train.txt')
    blocks_shape = np.array([block.shape[0] for block in blocks], dtype=np.int32)
    train_data = np.vstack(blocks)
    test_data= read_blocks('test.txt')

    blocks_label = np.genfromtxt('train_block_labels.txt', delimiter=' ', dtype=np.int32)



    train_labels = np.zeros((train_data.shape[0],))
    blocks_csum = np.insert(blocks_label.cumsum(), 0, 0)
    labels_count = 0
    for index in range(len(blocks_csum)-1):
        inc = blocks_shape[blocks_csum[index]:blocks_csum[index+1]].sum()
        train_labels[labels_count:labels_count+inc] = index
        #print(labels_count, inc, index)
        labels_count += inc

    return train_data, train_labels, test_data


def write_data(labels_test, filename):

    labels_test = list(enumerate(labels_test.ravel().astype(int)))

    np.savetxt(filename, labels_test,
               delimiter=',', fmt='%i', header='block_num, prediction', comments='')

    # with open('khan_speaker_labels_alternate.csv', 'wb') as fh:
    #     writer = csv.writer(fh, delimiter=',')
    #     writer.writerow(['block_num', 'prediction'])
    #     writer.writerows(labels_test)



def score_model(model, train_data):

    blocks_pred = np.empty((len(train_data),))
    blocks_consensus = np.empty((len(train_data),))
    for index, train_block in enumerate(train_data):
        y_pred = model.predict(train_block)
        if len(y_pred.shape) == 2:
            y_pred = np.argmax(y_pred, axis=1)
        blocks_pred[index] = stats.mode(y_pred)[0][0]
        blocks_consensus[index] = (stats.mode(y_pred)[1][0] / float(train_block.shape[0])) * 100

    return blocks_pred, blocks_consensus


def read_data_blocks():

    train_blocks = read_blocks('train.txt')
    test_blocks = read_blocks('test.txt')
    blocks_label = np.genfromtxt('train_block_labels.txt', delimiter=' ', dtype=np.int32)
    blocks_shape = np.array([block.shape[0] for block in train_blocks], dtype=np.int32)


    labels_count = 0
    train_blocks_labels = np.zeros((len(train_blocks),))
    blocks_csum = np.insert(blocks_label.cumsum(), 0, 0)
    train_labels = np.zeros((np.vstack(train_blocks).shape[0],))

    for index in range(len(blocks_csum)-1):
        train_blocks_labels[blocks_csum[index]:blocks_csum[index+1]] = index

        inc = blocks_shape[blocks_csum[index]:blocks_csum[index+1]].sum()
        train_labels[labels_count:labels_count+inc] = index
        labels_count += inc

    return train_blocks, train_blocks_labels, train_labels, test_blocks