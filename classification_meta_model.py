import sys
import pickle as pickle
from pprint import pprint
from functools import partial

from hyperband import Hyperband

# from defs.gb import get_params, try_params
# from defs.rf import get_params, try_params
# from defs.xt import get_params, try_params
# from defs.rf_xt import get_params, try_params
# from defs.sgd import get_params, try_params
# from defs.keras_mlp import get_params, try_params
# from defs.polylearn_fm import get_params, try_params
# from defs.polylearn_pn import get_params, try_params
# from defs.xgb import get_params, try_params
from defs.meta import get_params, try_params


def classification_meta_model(data, output_file='results.pkl'):
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'
    print("Will save results to", output_file)

    #

    try_params_data = partial(try_params, data=data)
    hb = Hyperband(get_params, try_params_data)
    results = hb.run(skip_last=1)

    print("{} total, best:\n".format(len(results)))

    for r in sorted(results, key=lambda x: x['loss'])[:5]:
        print("loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format(
            r['loss'], r['seconds'], r['iterations'], r['counter']))
        pprint(r['params'])
        print()

    print("saving...")

    with open(output_file, 'wb') as f:
        pickle.dump(results, f)