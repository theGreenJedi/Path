%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from IPython.core.pylabtools import figsize

iris_data=load_iris()  # Load the iris dataset
figsize(12.5, 10)
fig = plt.figure()
fig.suptitle('Plots of Iris Dimensions', fontsize=14)
fig.subplots_adjust(wspace=0.35,hspace=0.5)
colors=('r','g','b')
cols=[colors[i] for i in iris_data.target]

def get_legend_data(clrs):
    leg_data = []
    for clr in clrs:
        line=plt.Line2D(range(1),range(1),marker='o', color=clr)
        leg_data.append(line)
    return tuple(leg_data)


def display_iris_dimensions(fig,x_idx, y_idx,sp_idx):
    ax = fig.add_subplot(3,2,sp_idx)
    ax.scatter(iris_data.data[:, x_idx], iris_data.data[:,y_idx],c=cols)
    
    ax.set_xlabel(iris_data.feature_names[x_idx])
    ax.set_ylabel(iris_data.feature_names[y_idx])
    leg_data = get_legend_data(colors)
   
    ax.legend(leg_data,iris_data.target_names, numpoints=1,
              bbox_to_anchor=(1.265,1.0),prop={'size':8.5})
    
idx = 1
pairs = [(x,y) for x in range(0,4) for y in range(0,4) if x < y]
for (x,y) in pairs:
    display_iris_dimensions(fig,x,y,idx);
    idx += 1
    