from sklearn.datasets import load_sample_images
from matplotlib.pyplot import imshow, show, axis, plot
import numpy as np
from skimage.feature import harris

dataset = load_sample_images()
img = dataset.images[0] 
harris_coords = harris(img)
print "Harris coords shape", harris_coords.shape
y, x = np.transpose(harris_coords)
axis('off')
imshow(img)
plot(x, y, 'ro')
show()
