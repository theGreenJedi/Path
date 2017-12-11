%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

colors=['r','y','g','b']
def logistic(x, a):
    return 1. / (1. + np.exp(-a*x))
X = np.arange(-6, 6, .01)
pl_arr=[]
ax = plt.subplot(1,1,1)
for i in range(3):
    a=5**i
    Y = logistic(X,a)
    print Y
    plt.title('Logistic Function')
    pl,=ax.plot(X, Y,  color=colors[i], lw=2, label="a=%s" % a)
    ax.set_xlabel('x')
   

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
ax.annotate('0.5', xy=(0.2,0.5))