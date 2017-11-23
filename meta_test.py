from load_data_for_regression import data as data_reg
#from load_data import data as data_clf

from meta_models import regression_meta_model, classification_meta_model
from defs import keras_mlp

res = regression_meta_model(data_reg, 'try')
#res = classification_meta_model(data_clf, 'try')

print(res[0])