import matplotlib.pyplot as plt
import seaborn as sns
import config as cfg
import time, sys, os
from mlinsights.plotting import pipeline2dot, pipeline2str
from pyquickhelper.loghelper import run_cmd

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

def plot_pipeline(dot=None, pipeline=None, dataframe=None, name='pipeline'):
    
    dot_file = name + "_graph.dot"
    
    if dot == None:
        dot = pipeline2dot(pipeline, dataframe)
        with open(dot_file, "w", encoding="utf-8") as f:
            f.write(dot)

    if sys.platform.startswith("win") and "Graphviz" not in os.environ["PATH"]:
        os.environ['PATH'] = os.environ['PATH'] + r';C:\Program Files (x86)\Graphviz2.38\bin'

    cmd = "dot -G=300 -Tpng {0} -o{0}.png".format(dot_file)
    run_cmd(cmd, wait=True, fLOG=print);

    #img = Image.open("graph.dot.png")
    #img
