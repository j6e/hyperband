from defs import keras_mlp
from load_data_for_regression import data as data_reg
from meta_models import classification_meta_model, regression_meta_model

#from load_data import data as data_clf


res = regression_meta_model(data_reg, 'try')
#res = classification_meta_model(data_clf, 'try')

print(res[0])
