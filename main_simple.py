#!/usr/bin/env python

"bare-bones demonstration of using hyperband to tune sklearn GBT"

from defs.gb import get_params, try_params
from hyperband import Hyperband

hb = Hyperband(get_params, try_params)

# no actual tuning, doesn't call try_params()
# results = hb.run( dry_run = True )		

results = hb.run(skip_last=1)  # shorter run
results = hb.run()
