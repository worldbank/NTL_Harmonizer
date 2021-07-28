import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import numpy as np


def raster_scatter(x, y, xlabel, ylabel, title, opath, **kwargs):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    sns.scatterplot(x=x, y=y, ax=ax, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.legend(loc="best")
    if opath is not None:
        plt.savefig(opath)
    else:
        plt.show()
    plt.close(fig)


def raster_hist(
    x, y, bins, label1, label2, xlabel, ylabel, title, opath, text=None, **kwargs
):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    sns.histplot(
        x=x, ax=ax, color="indianred", alpha=0.5, bins=bins, label=label1, **kwargs
    )
    sns.histplot(
        x=y, ax=ax, color="steelblue", alpha=0.5, bins=bins, label=label2, **kwargs
    )
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=20)
    if text is not None:
        plt.text(bins // 3, len(x) // 10, text, ha="left", wrap=True)
    if opath is not None:
        plt.savefig(opath)
    else:
        plt.show()
    plt.close(fig)


def plot_timeseries(seqs, yrs, labels, opath, xlabel, ylabel, title, **kwargs):
    fig, ax = plt.subplots(1, figsize=(15, 7))
    for seq, yr, label in zip(seqs, yrs, labels):
        if "DMSP" in label:
            clr = "orange"
        else:
            clr = "b"
        if "un-adj" in label:
            sns.lineplot(x=yr, y=seq, ls=":", color=clr, label=label, ax=ax, **kwargs)
        else:
            sns.lineplot(x=yr, y=seq, color=clr, label=label, ax=ax, **kwargs)
    plt.legend(loc="best")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xticks(rotation=45)
    if opath is not None:
        plt.savefig(opath)
    else:
        plt.show()
    plt.close(fig)

def difference_plots(srcpath, thresh=1):
    with rasterio.open(srcpath) as src:
        arr = src.read(list(range(1, src.count + 1)))
        desc = src.descriptions
    for i in range(2):
        fig, ax = plt.subplots(1, figsize=(15, 7))
        plt.imshow(arr[i, ...])
        plt.title(f"2013 {desc[i]}", fontsize=20)
        plt.axis("off")

    fig, ax = plt.subplots(1, figsize=(15, 7))
    plt.imshow(np.ma.masked_array(arr, arr < thresh)[0, ...] - np.ma.masked_array(arr, arr < thresh)[1, ...], cmap='seismic')
    plt.colorbar()
    plt.title("2013 difference (VIIRS-DNB minus DMSP-OLS) (lit pixels only)", fontsize=20);
    plt.axis("off")