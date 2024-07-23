"""This optimization is based on a lumerical example using a Python API.
 The example is here : https://optics.ansys.com/hc/en-us/articles/360042800573-Inverse-Design-of-Grating-Coupler-2D""
"""

import numpy as np
from lumapi import FDTD
from lumopt.geometries.parameterized_geometry import ParameterizedGeometry
from lumopt.utilities.wavelengths import Wavelengths
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.optimization import Optimization

from ClassGEO import PhcForOptimization

#select which geometry you want to optimize by using this function, you have to replace the object of the classGEO
def run_phc_optimization(PhcForOptimization, initial_params: list, bounds: list):
    def gen_phc(params: list, fdtd: FDTD, only_update: bool):
        """This sub function allows the creation and the update of the geometry you want to optimize"""
        if not only_update:
            PhcForOptimization.setup_geometry(params, fdtd)
        else:
            PhcForOptimization.update_geometry(params, fdtd)

    geometry = ParameterizedGeometry(func=gen_phc, initial_params=initial_params, bounds=bounds, dx=1e-5,
                                     deps_num_threads=1)
    wavelengths = Wavelengths(0.93e-6, 0.945e-6, 25)

    fom = ModeMatch(monitor_name="fom",
                    mode_number="Fundamental TE mode",
                    direction="Forward",
                    target_T_fwd=lambda wl: np.ones(wl.size),
                    norm_p=1,
                    target_fom=1.)

    optimizer = ScipyOptimizers(max_iter=250, method='L-BFGS-B', scaling_factor=1, ftol=1e-6, pgtol=1e-6)
    opt = Optimization(base_script=PhcForOptimization.setup_script,
                       wavelengths=wavelengths,
                       fom=fom,
                       geometry=geometry,
                       optimizer=optimizer,
                       hide_fdtd_cad=False,
                       use_deps=True)

    return opt.run(working_dir="./optim_export")
