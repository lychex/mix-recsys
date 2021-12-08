import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os
import random
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PROJECT_DIR = os.getcwd()
IMAGES_PATH = os.path.join(PROJECT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_name, image_path=IMAGES_PATH, tight_layout=False, fig_extension="png", resolution=300):
    path = os.path.join(image_path, fig_name + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension,
                dpi=resolution, bbox_inches="tight")
    print("Figure saved,", fig_name)


def plot_piechart(col_name, data, explode=None):
    plt.style.use("ggplot")
    plt.rcParams["figure.figsize"] = [7, 7]
    plt.rcParams["figure.autolayout"] = True
    n = data[col_name].nunique()
    slices = data[col_name].value_counts().values
    activities = data[col_name].value_counts().index
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF')
                            for j in range(6)]) for i in range(n)]

    patches, texts = plt.pie(slices,
                             colors=colors,
                             startangle=90,
                             labels=activities, explode=explode)

    labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(activities,
                                                              100.*slices/slices.sum())]
    plt.legend(patches, labels, loc='center left',
               bbox_to_anchor=(-0.7, .5), fontsize=8)

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    f = plt.gcf()
    f.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()


def plot_distribution(data, label=""):
    plt.style.use('default')
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1.05, 1])
    sns.distplot(data, ax=ax)
    plt.axvline(np.mean(data), color="k", linestyle="dashed", linewidth=2)
    _, max_ = plt.ylim()
    plt.text(
        np.mean(data) + np.mean(data) / 10,
        max_ - max_ / 10,
        "{0} Mean: {1:.2f}".format(label, np.mean(data)),)


def plot_clusters(clusterer_name, clusterer, X):
    """
    Data features will be reduced to two dimensions for visualization, 
    PCA is used for dimensionality reduction,
    X = unsupervised dataframe i.e., without label column
    """
    preprocessor = Pipeline(
        [("scaler", MinMaxScaler()), ("pca", PCA(n_components=2, random_state=12))])
    clusterer = Pipeline([(clusterer_name, clusterer)])
    pipe = Pipeline([("preprocessor", preprocessor), ("clusterer", clusterer)])
    X_preprocessed = pipe["preprocessor"].fit_transform(X)
    pipe.fit(X)
    try:
        predicted_labels = pipe["clusterer"][clusterer_name].labels_
    except:
        predicted_labels = pipe["clusterer"][clusterer_name].predict(
            X_preprocessed)
    print(
        f"Silhouette Score:{silhouette_score(X_preprocessed, predicted_labels).round(4)}")
    pca_df = pd.DataFrame(pipe["preprocessor"].transform(
        X), columns=["component_1", "component_2"], )
    pca_df["predicted_cluster"] = predicted_labels

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(4, 4))

    sns.scatterplot(x="component_1", y="component_2", s=30, data=pca_df,
                    hue="predicted_cluster", palette="Set2", ax=ax)
    ax.set_title(f"{clusterer_name.upper()} Clustering")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)


def plot_tsne(X):
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(
        X), index=X.index, columns=X.columns)

    # perplixity specification
    pers = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    fig, ax = plt.subplots(3, 3, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    for i, perplexity in enumerate(pers):
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=214)
        X_embeded = tsne.fit_transform(X_scaled)
        tsne_results = pd.DataFrame(X_embeded, index=X_scaled.index)

        ax[i % 3][math.floor(i/3)].set_title(f"perplexity {perplexity}")
        ax[i % 3][math.floor(i/3)].scatter(x=tsne_results[0],
                                           y=tsne_results[1], s=10, alpha=0.2)
        ax[i % 3][math.floor(i/3)].set(xlabel='TSNE1')
        ax[i % 3][0].set(ylabel='TSNE2')


def plot_dbscan(tsne_results):
    # Tune sample size and eps
    sample_list = [50, 80, 110]
    eps_list = [7, 7.5, 8, 8.5, 9]
    rows = len(sample_list)

    fig, ax = plt.subplots(len(sample_list), len(
        eps_list), figsize=(6*len(sample_list), 2*len(eps_list)))

    for k, (eps, samp) in enumerate(it.product(eps_list, sample_list)):
        clusterer = DBSCAN(eps=eps, min_samples=samp).fit(tsne_results)
        tsne_df = tsne_results.copy()
        tsne_df.insert(0, 'cluster', clusterer.labels_)

        n = len(set(clusterer.labels_))
        data_list = [[tsne_df[tsne_df['cluster'] == i][j]
                      for j in range(2)] for i in range(n)]
        random.seed(99)
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF')
                                for j in range(6)]) for i in range(n)]
        groups = [str(i) for i in range(n)]

        for data, color, group in zip(data_list, colors, groups):
            ax[k % rows][math.floor(k/rows)].scatter(data[0],
                                                     data[1], alpha=0.2, c=color, label=group)
            ax[k % rows][math.floor(
                k/rows)].set_title(f"DBS eps: {eps}, sample: {samp}")
