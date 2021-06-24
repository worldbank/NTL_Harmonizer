import matplotlib.pyplot as plt
import seaborn as sns


def raster_scatter(x, y, xlabel, ylabel, title, opath, **kwargs):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    sns.scatterplot(x=x, y=y, ax=ax, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    if opath is not None:
        plt.savefig(opath)
    else:
        plt.show()
    plt.close(fig)


def raster_hist(
    x, y, bins, label1, label2, xlabel, ylabel, title, opath, text, **kwargs
):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    sns.histplot(
        x=x, ax=ax, color="indianred", alpha=0.5, bins=bins, label=label1, **kwargs
    )
    sns.histplot(
        x=y, ax=ax, color="steelblue", alpha=0.5, bins=bins, label=label2, **kwargs
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
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
        if "comp" in label:
            sns.lineplot(x=yr, y=seq, ls=":", label=label, ax=ax, **kwargs)
        else:
            sns.lineplot(x=yr, y=seq, label=label, ax=ax, **kwargs)
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
