import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg
import time

def my_fig(df):
    fig, ax = plt.subplots()
    df.plot(kind='hist', ax=ax)
    fig.savefig(cfg.OUTPUT_PATH+'plot_'+ time.strftime('%Y%m%d-%H%M%S') + '.png')

def missing_values_heatmap(df):
    fig, ax = plt.subplots(figsize=(30,12))
    sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='plasma',ax=ax)
    ax.set_title('Missing Values Heatmap', fontdict=cfg.font)
    fig.savefig(cfg.OUTPUT_PATH+'missing-heatmap_'+ time.strftime('%Y%m%d-%H%M%S') + '.png')

def cor_matrix(df):
    fig, ax = plt.subplots(figsize=(30,12))

    # Compute the correlation matrix
    ad -y /tmp/nvimwtxgr1/4.ipy

    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, 
            mask=mask, 
            cmap=cmap, 
            vmax=.3, 
            center=0, 
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5},
            ax=ax)

    ax.set_title('Correlation Matrix', fontdict=cfg.font)
    fig.savefig(cfg.OUTPUT_PATH+'cor-matrix_'+ time.strftime('%Y%m%d-%H%M%S') + '.png')
#
