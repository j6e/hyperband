import pickle as pickle
from pprint import pprint
from functools import partial

from hyperband import Hyperband
from defs.meta import get_params as get_params_c
from defs.meta import try_params as try_params_c
from defs_regression.meta import get_params as get_params_r
from defs_regression.meta import try_params as try_params_r


def classification_meta_model(data, output_file='results.pkl', max_iter=81, eta=3):
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'
    print("Will save results to", output_file)

    #

    try_params_data = partial(try_params_c, data=data)
    hb = Hyperband(get_params_c, try_params_data, max_iter=max_iter, eta=3)
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

    return results


def regression_meta_model(data, output_file='results.pkl', max_iter=81, eta=3):
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'
    print("Will save results to", output_file)

    #
    try_params_data = partial(try_params_r, data=data)

    hb = Hyperband(get_params_r, try_params_data, max_iter=max_iter, eta=eta)
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

    return results
