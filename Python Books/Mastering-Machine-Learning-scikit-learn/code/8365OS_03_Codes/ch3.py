__author__ = 'gavin'
"""

"""
################# Sample 1 #################
"""
>>> from sklearn.feature_extraction import DictVectorizer
>>> onehot_encoder = DictVectorizer()
>>> instances = [
>>>     {'city': 'New York'},
>>>     {'city': 'San Francisco'},
>>>     {'city': 'Chapel Hill'}
>>> ]
>>> print onehot_encoder.fit_transform(instances).toarray()
[[ 0.  1.  0.]
 [ 0.  0.  1.]
 [ 1.  0.  0.]]
"""
from sklearn.feature_extraction import DictVectorizer
onehot_encoder = DictVectorizer()
instances = [
    {'city': 'New York'},
    {'city': 'San Francisco'},
    {'city': 'Chapel Hill'}
]
print onehot_encoder.fit_transform(instances).toarray()


################# Sample 2 #################
"""
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]
"""
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]

################# Sample 3 #################
"""
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
>>>     'UNC played Duke in basketball',
>>>     'Duke lost the basketball game'
>>> ]
>>> vectorizer = CountVectorizer(binary=True)
>>> print vectorizer.fit_transform(corpus).todense()
>>> print vectorizer.vocabulary_
[[1 1 0 1 0 1 0 1]
 [1 1 1 0 1 0 1 0]]
{u'duke': 1, u'basketball': 0, u'lost': 4, u'played': 5, u'game': 2, u'unc': 7, u'in': 3, u'the': 6}
"""
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game'
]
vectorizer = CountVectorizer(binary=True)
print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_


################# Sample 4 #################
"""
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
>>>     'UNC played Duke in basketball',
>>>     'Duke lost the basketball game',
>>>     'I ate a sandwich'
>>> ]
>>> vectorizer = CountVectorizer(binary=True)
>>> X = vectorizer.fit_transform(corpus).todense()
>>> print X
>>> print vectorizer.vocabulary_
>>> for i, document in enumerate(corpus):
>>>     print document, '=', X[i]
{u'duke': 2, u'basketball': 1, u'lost': 5, u'played': 6, u'in': 4, u'game': 3, u'sandwich': 7, u'unc': 9, u'ate': 0, u'the': 8}
UNC played Duke in basketball = [[0 1 1 0 1 0 1 0 0 1]]
Duke lost the basketball game = [[0 1 1 1 0 1 0 0 1 0]]
I ate a sandwich = [[1 0 0 0 0 0 0 1 0 0]]
"""
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(corpus).todense()
print X
print vectorizer.vocabulary_
for i, document in enumerate(corpus):
    print document, '=', X[i]


################# Sample 6 #################
"""
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
>>>     'UNC played Duke in basketball',
>>>     'Duke lost the basketball game',
>>>     'I ate a sandwich'
>>> ]
>>> vectorizer = CountVectorizer(binary=True, stop_words='english')
>>> print vectorizer.fit_transform(corpus).todense()
>>> print vectorizer.vocabulary_
[[0 1 1 0 0 1 0 1]
 [0 1 1 1 1 0 0 0]
 [1 0 0 0 0 0 1 0]]
{u'duke': 2, u'basketball': 1, u'lost': 4, u'played': 5, u'game': 3, u'sandwich': 6, u'unc': 7, u'ate': 0}
"""
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'UNC played Duke in basketball',
    'Duke lost the basketball game',
    'I ate a sandwich'
]
vectorizer = CountVectorizer(binary=True, stop_words='english')
print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_


################# Sample 7 #################
"""
>>> from sklearn.metrics.pairwise import euclidean_distances
>>> counts = [
>>>     [0, 1, 1, 0, 0, 1, 0, 1],
>>>     [0, 1, 1, 1, 1, 0, 0, 0],
>>>     [1, 0, 0, 0, 0, 0, 1, 0]
>>> ]
>>> print 'Distance between 1st and 2nd documents:', euclidean_distances(counts[0], counts[1])
>>> print 'Distance between 1st and 3rd documents:', euclidean_distances(counts[0], counts[2])
>>> print 'Distance between 2nd and 3rd documents:', euclidean_distances(counts[1], counts[2])
Distance between 1st and 2nd documents: [[ 2.]]
Distance between 1st and 3rd documents: [[ 2.44948974]]
Distance between 2nd and 3rd documents: [[ 2.44948974]]
"""
from sklearn.metrics.pairwise import euclidean_distances
counts = [
    [0, 1, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0]
]
print 'Distance between 1st and 2nd documents:', euclidean_distances(counts[0], counts[1])
print 'Distance between 1st and 3rd documents:', euclidean_distances(counts[0], counts[2])
print 'Distance between 2nd and 3rd documents:', euclidean_distances(counts[1], counts[2])

################# Sample 8 #################
"""
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = [
>>>     'He ate the sandwiches',
>>>     'Every sandwich was eaten by him'
>>> ]
>>> vectorizer = CountVectorizer(binary=True, stop_words='english')
>>> print vectorizer.fit_transform(corpus).todense()
>>> print vectorizer.vocabulary_
[[1 0 0 1]
 [0 1 1 0]]
{u'sandwich': 2, u'ate': 0, u'sandwiches': 3, u'eaten': 1}
"""
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]
vectorizer = CountVectorizer(binary=True, stop_words='english')
print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_


################# Sample 7 #################
"""
corpus = [
    'I am gathering ingredients for the sandwich.',
    'There were many wizards at the gathering.'
]
"""


################# Sample 8 #################
"""
>>> from nltk.stem.wordnet import WordNetLemmatizer
>>> lemmatizer = WordNetLemmatizer()
>>> print lemmatizer.lemmatize('gathering', 'v')
>>> print lemmatizer.lemmatize('gathering', 'n')
gather
gathering
"""
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print lemmatizer.lemmatize('gathering', 'v')
print lemmatizer.lemmatize('gathering', 'n')


################# Sample 8 #################
"""
>>> from nltk.stem import PorterStemmer
>>> stemmer = PorterStemmer()
>>> print stemmer.stem('gathering')
gather
"""
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print stemmer.stem('gathering')


################# Sample 9 #################
"""
>>> from nltk import word_tokenize
>>> from nltk.stem import PorterStemmer
>>> from nltk.stem.wordnet import WordNetLemmatizer
>>> from nltk import pos_tag
>>> wordnet_tags = ['n', 'v']
>>> corpus = [
>>>     'He ate the sandwiches',
>>>     'Every sandwich was eaten by him'
>>> ]

>>> stemmer = PorterStemmer()
>>> print 'Stemmed:', [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus]


>>> def lemmatize(token, tag):
>>>     if tag[0].lower() in ['n', 'v']:
>>>         return lemmatizer.lemmatize(token, tag[0].lower())
>>>     return token

>>> lemmatizer = WordNetLemmatizer()
>>> tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
>>> print 'Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus]
Stemmed: [['He', 'ate', 'the', 'sandwich'], ['Everi', 'sandwich', 'wa', 'eaten', 'by', 'him']]
Lemmatized: [['He', 'eat', 'the', 'sandwich'], ['Every', 'sandwich', 'be', 'eat', 'by', 'him']]
"""
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
wordnet_tags = ['n', 'v']
corpus = [
    'He ate the sandwiches',
    'Every sandwich was eaten by him'
]

stemmer = PorterStemmer()
print 'Stemmed:', [[stemmer.stem(token) for token in word_tokenize(document)] for document in corpus]


def lemmatize(token, tag):
    if tag[0].lower() in ['n', 'v']:
        return lemmatizer.lemmatize(token, tag[0].lower())
    return token

lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(document)) for document in corpus]
print 'Lemmatized:', [[lemmatize(token, tag) for token, tag in document] for document in tagged_corpus]


################# Sample 10 #################
"""
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
>>> vectorizer = CountVectorizer(stop_words='english')
>>> print vectorizer.fit_transform(corpus).todense()
[[2 1 3 1 1]]
{u'sandwich': 2, u'wizard': 4, u'dog': 1, u'transfigured': 3, u'ate': 0}
"""
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['The dog ate a sandwich, the wizard transfigured a sandwich, and I ate a sandwich']
vectorizer = CountVectorizer(stop_words='english')
print vectorizer.fit_transform(corpus).todense()
print vectorizer.vocabulary_


################# Sample 10 #################
"""

"""
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'The dog foo bar dog dog dog dog foo bar',
    'Dog the hat'
]
vectorizer = CountVectorizer(stop_words='english')
print vectorizer.fit_transform(corpus).todense()


################# Sample 11 #################
"""
>>> from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
>>> corpus = [
>>>     'The dog ate a sandwich',
>>>     'The wizard transfigured a sandwich',
>>>     'I ate a sandwich'
>>> ]
>>> vectorizer = CountVectorizer(stop_words='english')
>>> transformer = TfidfTransformer()
>>> X = vectorizer.fit_transform(corpus)
>>> print vectorizer.vocabulary_
>>> print transformer.fit_transform(X).todense()
{u'sandwich': 2, u'wizard': 4, u'dog': 1, u'transfigured': 3, u'ate': 0}
[[ 0.54783215  0.72033345  0.42544054  0.          0.        ]
 [ 0.          0.          0.38537163  0.65249088  0.65249088]
 [ 0.78980693  0.          0.61335554  0.          0.        ]]
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
corpus = [
    'The dog ate a sandwich and I ate a sandwich',
    'The wizard transfigured a sandwich'
]
vectorizer = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(use_idf=False)
X = vectorizer.fit_transform(corpus)
print 'Count vectors:\n', X.todense()
print 'Vocabulary:\n', vectorizer.vocabulary_
print 'TF vectors:\n', transformer.fit_transform(X).todense()


################# Sample 12 #################
"""
>>> from sklearn.feature_extraction.text import TfidfVectorizer
>>> corpus = [
>>>     'The dog ate a sandwich and I ate a sandwich',
>>>     'The wizard transfigured a sandwich'
>>> ]
>>> vectorizer = TfidfVectorizer(stop_words='english')
>>> print vectorizer.fit_transform(corpus).todense()
[[ 0.75458397  0.37729199  0.53689271  0.          0.        ]
 [ 0.          0.          0.44943642  0.6316672   0.6316672 ]]
"""
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'The dog ate a sandwich and I ate a sandwich',
    'The wizard transfigured a sandwich'
]
vectorizer = TfidfVectorizer(stop_words='english')
print vectorizer.fit_transform(corpus)


################# Sample 13 #################
"""
>>> from sklearn.feature_extraction.text import HashingVectorizer
>>> corpus = ['the', 'ate', 'bacon', 'cat']
>>> vectorizer = HashingVectorizer(n_features=6)
>>> print vectorizer.transform(corpus).todense()
[[-1.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  1.  0.  0.]
 [ 0.  0.  0.  0. -1.  0.]
 [ 0.  1.  0.  0.  0.  0.]]
"""
from sklearn.feature_extraction.text import HashingVectorizer
corpus = ['the', 'ate', 'bacon', 'cat']
vectorizer = HashingVectorizer(n_features=6)
print vectorizer.transform(corpus).todense()


################# Figure 14 #################
"""

"""
from sklearn import datasets
import matplotlib.pyplot as plt
digits = datasets.load_digits()
print 'Digit:', digits.target[0]
print digits.images[0]
plt.figure()
plt.axis('off')
plt.imshow(digits.images[0], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

################# Sample 14 #################
"""
>>> from sklearn import datasets
>>> digits = datasets.load_digits()
>>> print 'Digit:', digits.target[0]
>>> print digits.images[0]
>>> print 'Feature vector:\n', digits.images[0].reshape(-1, 64)
Digit: 0
[[  0.   0.   5.  13.   9.   1.   0.   0.]
 [  0.   0.  13.  15.  10.  15.   5.   0.]
 [  0.   3.  15.   2.   0.  11.   8.   0.]
 [  0.   4.  12.   0.   0.   8.   8.   0.]
 [  0.   5.   8.   0.   0.   9.   8.   0.]
 [  0.   4.  11.   0.   1.  12.   7.   0.]
 [  0.   2.  14.   5.  10.  12.   0.   0.]
 [  0.   0.   6.  13.  10.   0.   0.   0.]]
Feature vector:
[[  0.   0.   5.  13.   9.   1.   0.   0.   0.   0.  13.  15.  10.  15.
    5.   0.   0.   3.  15.   2.   0.  11.   8.   0.   0.   4.  12.   0.
    0.   8.   8.   0.   0.   5.   8.   0.   0.   9.   8.   0.   0.   4.
   11.   0.   1.  12.   7.   0.   0.   2.  14.   5.  10.  12.   0.   0.
    0.   0.   6.  13.  10.   0.   0.   0.]]
"""
from sklearn import datasets
digits = datasets.load_digits()
print 'Digit:', digits.target[0]
print digits.images[0]
print 'Feature vector:\n', digits.images[0].reshape(-1, 64)

################# Sample 15 #################
"""
>>> import numpy as np
>>> from skimage.feature import corner_harris, corner_peaks
>>> from skimage.color import rgb2gray
>>> import matplotlib.pyplot as plt
>>> import skimage.io as io
>>> from skimage.exposure import equalize_hist

>>> def show_corners(corners, image):
>>>     fig = plt.figure()
>>>     plt.gray()
>>>     plt.imshow(image)
>>>     y_corner, x_corner = zip(*corners)
>>>     plt.plot(x_corner, y_corner, 'or')
>>>     plt.xlim(0, image.shape[1])
>>>     plt.ylim(image.shape[0], 0)
>>>     fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
>>>     plt.show()

>>> mandrill = io.imread('mandrill.png')
>>> mandrill = equalize_hist(rgb2gray(mandrill))
>>> corners = corner_peaks(corner_harris(mandrill), min_distance=2)
>>> show_corners(corners, mandrill)
"""
import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.exposure import equalize_hist


def show_corners(corners, image):
    fig = plt.figure()
    plt.gray()
    plt.imshow(image)
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, 'or')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()

mandrill = io.imread('mandrill.png')
mandrill = equalize_hist(rgb2gray(mandrill))
corners = corner_peaks(corner_harris(mandrill), min_distance=2)
show_corners(corners, mandrill)


################# Sample 16 #################
"""
>>> import mahotas as mh
>>> from mahotas.features import surf

>>> image = mh.imread('data/zipper.jpg', as_grey=True)
>>> print 'The first SURF descriptor:\n', surf.surf(image)[0]
>>> print 'Extracted %s SURF descriptors' % len(surf.surf(image))
The first SURF descriptor:
[  6.73839947e+02   2.24033945e+03   3.18074483e+00   2.76324459e+03
  -1.00000000e+00   1.61191475e+00   4.44035121e-05   3.28041690e-04
   2.44845817e-04   3.86297608e-04  -1.16723672e-03  -8.81290243e-04
   1.65414959e-03   1.28393061e-03  -7.45077384e-04   7.77655540e-04
   1.16078772e-03   1.81434398e-03   1.81736394e-04  -3.13096961e-04
   3.06559785e-04   3.43443699e-04   2.66200498e-04  -5.79522387e-04
   1.17893036e-03   1.99547411e-03  -2.25938217e-01  -1.85563853e-01
   2.27973631e-01   1.91510135e-01  -2.49315698e-01   1.95451021e-01
   2.59719480e-01   1.98613061e-01  -7.82458546e-04   1.40287015e-03
   2.86712113e-03   3.15971628e-03   4.98444730e-04  -6.93986983e-04
   1.87531652e-03   2.19041521e-03   1.80681053e-01  -2.70528820e-01
   2.32414943e-01   2.72932870e-01   2.65725332e-01   3.28050743e-01
   2.98609869e-01   3.41623138e-01   1.58078002e-03  -4.67968721e-04
   2.35704122e-03   2.26279888e-03   6.43115065e-06   1.22501486e-04
   1.20064616e-04   1.76564805e-04   2.14148537e-03   8.36243899e-05
   2.93382280e-03   3.10877776e-03   4.53469215e-03  -3.15254535e-04
   6.92437341e-03   3.56880279e-03  -1.95228401e-04   3.73674995e-05
   7.02700555e-04   5.45156362e-04]
Extracted 994 SURF descriptors
"""
import mahotas as mh
from mahotas.features import surf

image = mh.imread('data/zipper.jpg', as_grey=True)
print 'The first SURF descriptor:\n', surf.surf(image)[0]
print 'Extracted %s SURF descriptors' % len(surf.surf(image))


################# Sample 17 #################
"""
>>> from sklearn import preprocessing
>>> import numpy as np
>>> X = np.array([
>>>     [0., 0., 5., 13., 9., 1.],
>>>     [0., 0., 13., 15., 10., 15.],
>>>     [0., 3., 15., 2., 0., 11.]
>>> ])
>>> print preprocessing.scale(X)
[[ 0.         -0.70710678 -1.38873015  0.52489066  0.59299945 -1.35873244]
 [ 0.         -0.70710678  0.46291005  0.87481777  0.81537425  1.01904933]
 [ 0.          1.41421356  0.9258201  -1.39970842 -1.4083737   0.33968311]]
"""
from sklearn import preprocessing
import numpy as np
X = np.array([
    [0., 0., 5., 13., 9., 1.],
    [0., 0., 13., 15., 10., 15.],
    [0., 3., 15., 2., 0., 11.]
])
print preprocessing.scale(X)
