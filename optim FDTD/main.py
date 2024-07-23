from ClassGEO import Phc_Paper
from ClassGEO import Phc
import numpy as np
from optimization import run_phc_optimization

if __name__ == '__main__':
    p0 = [0]
    bounds = [(p - 2, p + 2) for p in p0]

    #geom = Phc()
    # run_phc_optimization(geom,p0,bounds)

    geom1=Phc_Paper()
    geom1.check_setup(p0)

