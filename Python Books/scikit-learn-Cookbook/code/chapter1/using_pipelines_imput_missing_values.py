# IPython log file


from __future__ import division
import numpy as np
from sklearn import datasets
get_ipython().set_next_input(u'mat = datasets.make_spd_matrix');get_ipython().magic(u'pinfo datasets.make_spd_matrix')
mat = datasets.make_spd_matrix(10)
mat
mat.mean()
mat.mean(axis=1)
mat.mean(axis=0)
masking_array = np.random.binomial(1, .1, mat.shape).astype(bool)
mat[masking_array] = np.nan
mat
mat[:4, :4]
from sklearn import preprocessing
impute = preprocessing.Imputer()
scalar = preprocessing.StandardScaler()
mat_imputed = impute.fit_transform(mat)
mat_imputed[:4, :4]
mat_imp_and_scaled = scalar.fit_transform(mat_imputed)
mat_imp_and_scaled[:4, :4]
np.set_printoptions(precision=3)
mat_imp_and_scaled[:4, :4]
from sklearn import pipeline
pipe = pipeline.Pipeline()
pipe = pipeline.Pipeline([('impute', impute), ('scalar', scalar)])
pipe
pipe.fit_transform(mat)
np.array_equal(mat_imp_and_scaled, pipe.fit_transform(mat))
new_mat = pipe.fit_transform(mat)
new_mat[:4, :4]
pipe.inverse_transform(new_mat)
scalar.inverse_transform(new_mat)
scalar.inverse_transform(new_mat)[:4, :4]
exit()
