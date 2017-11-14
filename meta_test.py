from load_data_for_regression import data

from meta_models import regression_meta_model

res = regression_meta_model(data, 'try')

print(res[0])