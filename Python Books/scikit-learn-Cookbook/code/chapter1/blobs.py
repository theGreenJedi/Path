import sklearn.datasets as d
from matplotlib import pyplot as plt
import numpy as np

blobs = d.make_blobs(200)

f = plt.figure(figsize=(8, 4))


ax = f.add_subplot(111)
ax.set_title("A blob with 3 centers.")

colors = np.array(['r', 'g', 'b'])
ax.scatter(blobs[0][:, 0], blobs[0][:, 1], color=colors[blobs[1].astype(int)],
            alpha=0.75)

f.savefig("blobs.png")
