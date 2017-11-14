"A regression example. Mostly the same, only importing from defs_regression."

import sys
import pickle as pickle
from pprint import pprint
from functools import partial

from hyperband import Hyperband
from defs_regression.meta import get_params, try_params


def regression_meta_model(data, output_file):
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
