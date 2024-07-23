import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict

blabla
from lumapi import FDTD

class GeomBase(ABC):
    @abstractmethod
    def setup_script(self, fdtd: FDTD):
        """ Sets up the base simulation """
        pass

    @abstractmethod
    def setup_geometry(self, params: list, fdtd: FDTD):
        """ Sets up the optimizable geometry """
        pass

    @abstractmethod
    def update_geometry(self, params: list, fdtd: FDTD):
        """ Updates the optimizable geometry """
        pass

    def check_setup(self, params: list, filename=None):
        if filename is None:
            filename = f"{self.__class__.__name__}_geometry.fsp"

        fdtd = FDTD()
        self.setup_script(fdtd)
        self.setup_geometry(params, fdtd)
        fdtd.save(filename)

        return fdtd

class Phc(GeomBase):
    def __init__(self):
        self.wg_region_width = 1.5e-6
        self.wg_width = 0.6e-6
        self.wg_region_height = 0.173e-6
        self.n_wg = 3.46

        self.membrane_length = 22.3844
        self.membrane_width = 11.8174
        self.trench_width = 1.0
        self.wg_z_span = 0.176376
        self.wg1_width = 0.397017

        self.x_span = 2 * 8.69221e-6
        self.y_span = 8e-6
        self.z_span = 2e-6

        self.ny_fr = 10
        self.hole_radius = 0.077e-6
        self.a = 0.246
        self.d_fregion = 2.46
        self.lambda0 = 0.94e-6
        self.lambda_span = 0.02e-6

        self.dx = 0.03e-6
        self.dy = 0.03e-6
        self.dz = 0.013e-6

        self._source_pos_x = -self.x_span / 2
        self._fom_pos_x = self.x_span / 2

        self._mode_span_y = 1.5e-6
        self._mode_span_z = 0.881e-6

    def setup_script(self, fdtd: FDTD):

        filename = "C:\\Users\\SQ\\Documents\\Lumerical\\Photonic_crystals\\FSF_crystal\\04_optimization_python\\Phc_slow_without_wg.fsp"

        fdtd.load(filename)

        # props = OrderedDict([
        #     ("dimension", "3D"),
        #     ("x", 0.),
        #     ("x span", self.x_span),
        #     ("y", 0.),
        #     ("y span", self.y_span),
        #     ("z", 0.),
        #     ("z span", self.z_span),
        #     ("y min bc", "anti-symmetric"),
        #     ("mesh accuracy", 2),
        #     ("simulation time", 10000e-15),
        #     ("auto shutoff min", 1e-3),
        # ])
        #
        # fdtd.addfdtd(properties=props)

        # fdtd.addrect(name="membrane",
        #              x=0, y=0, z=0,
        #              x_span=self.membrane_length * 1e-6,
        #              y_span=self.membrane_width * 1e-6,
        #              z_span=self.wg_region_height,
        #              index=self.n_wg)
        #
        # fdtd.addrect(name="inner membrane",
        #              x=0, y=0, z=0.0316e-6,
        #              x_span=self.membrane_length * 1e-6,
        #              y_span=self.membrane_width * 1e-6,
        #              z_span=0.047e-6,
        #              index=self.n_wg)

        fdtd.addmode(name="source",
                     injection_axis="x-axis",
                     direction="forward",
                     x=self._source_pos_x,
                     y=0, z=0,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     mode_selection="fundamental TE mode",
                     center_wavelength=self.lambda0,
                     wavelength_span=self.lambda_span)

        fdtd.setglobalsource("center wavelength", self.lambda0)
        fdtd.setglobalsource("wavelength span", self.lambda_span)
        fdtd.setglobalmonitor("frequency points", 100)

        fdtd.addpower(name="fom",
                      monitor_type="2D X-normal",
                      x=self._fom_pos_x,
                      y=0, z=0,
                      y_span=self._mode_span_y,
                      z_span=self._mode_span_z,
                      override_global_monitor_settings=True,
                      frequency_points=100
                      )

        fdtd.addmesh(name="fom_mesh",
                     x=self._fom_pos_x,
                     y=0, z=0,
                     x_span=2 * self.dx,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     override_x_mesh=True,
                     override_y_mesh=False,
                     override_z_mesh=False,
                     dx=self.dx,
                     )

        fdtd.addpower(name="opt_fields",
                      monitor_type="2D Z-normal",
                      x=0, y=0, z=0,
                      x_span=self.x_span,
                      y_span=self._mode_span_y,
                      override_global_monitor_settings=True,
                      frequency_points=20,
                      spatial_interpolation="specified position",
                      record_data_in_pml=True
                      )

        fdtd.addmesh(name="opt_fields_mesh",
                     x=0, y=0, z=0,
                     x_span=self.x_span,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     dx=self.dx, dy=self.dy, dz=self.dz)

        fdtd.redrawon()

    def setup_geometry(self, params: list, fdtd: FDTD):

        (x_coords_right, y_coords_right), (x_coords_left, y_coords_left) = self._create_fast_regions_air(params)
        (x_min_coords, y_min_coords, x_max_coords, y_max_coords) = self._create_wg(params)

        # Adding rect for WG
        for i, (x_min, y_min, x_max, y_max) in enumerate(zip(x_min_coords, y_min_coords, x_max_coords, y_max_coords)):
            fdtd.addrect(name=f'rect{i}', x_min=x_min * 1e-6, y_min=y_min * 1e-6, x_max=x_max * 1e-6,
                         y_max=y_max * 1e-6, z=0, z_span=0.176376e-6, index=1.4, material="etch")

        # Adding circles for the right region
        for i, (x, y) in enumerate(zip(x_coords_right, y_coords_right)):
            fdtd.addcircle(name=f'circle_right_{i}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                           radius=self.hole_radius, index=1.4, material="etch")

        # Adding circles for the left region
        for j, (x, y) in enumerate(zip(x_coords_left, y_coords_left)):
            fdtd.addcircle(name=f'circle_left_{j}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                           radius=self.hole_radius, index=1.4, material="etch")

    def update_geometry(self, params: list, fdtd: FDTD):

        (x_coords_right, y_coords_right), (x_coords_left, y_coords_left) = self._create_fast_regions_air(params)
        (x_min_coords, y_min_coords, x_max_coords, y_max_coords) = self._create_wg(params)

        for i, (x, y) in enumerate(zip(x_coords_right, y_coords_right)):
            coordinates = {"x": x * 1e-6,
                           "y": y * 1e-6,
                           "z": 0,
                           "z span": 0.176376e-6,
                           }
            fdtd.setnamed(f'circle_right_{i}', coordinates)

        for i, (x, y) in enumerate(zip(x_coords_left, y_coords_left)):
            coordinates = {"x": x * 1e-6,
                           "y": y * 1e-6,
                           "z": 0,
                           "z span": 0.176376e-6,
                           "radius": self.hole_radius
                           }
            fdtd.setnamed(f'circle_left_{i}', coordinates)

        for i, (x_min, y_min, x_max, y_max) in enumerate(zip(x_min_coords, y_min_coords, x_max_coords, y_max_coords)):
            coordinatesr = {"x min": x_min * 1e-6,
                            "x max": x_max * 1e-6,
                            "y min": y_min * 1e-6,
                            "y max": y_max * 1e-6,
                            "z": 0,
                            "z span": 0.176376e-6,
                            }
            fdtd.setnamed(f'rect{i}', coordinatesr)

    def _create_fast_regions_air(self, params):
        ny_fr = 10
        n_col = len(params)
        sf1 = 1.01383
        sf2 = 1.09038
        a_poly = 7.79169e-05
        n_poly = 2 * n_col
        b_poly = (sf1 - sf2) / (1 - n_poly) - a_poly * (1 + n_poly)
        c_poly = sf1 - a_poly * sf1 ** 2 - b_poly * sf1
        sf_vals = [a_poly * i ** 2 + b_poly * i + c_poly for i in range(1, n_poly + 1)]
        x_coords_right = []
        y_coords_right = []
        x_coords_left = []
        y_coords_left = []
        tab_space1 = np.array(params)
        tab_space2 = np.array([sf1])
        for i in range(1, len(params)):
            tab_space2 = np.append(tab_space2, (params[i - 1] + params[i]) / 2)

        combined_array = np.concatenate((tab_space1, tab_space2))
        sorted_array = np.sort(combined_array)
        cumul = np.cumsum(sorted_array)
        for i in range(-self.ny_fr, self.ny_fr + 1):
            if i == 0:
                continue
            for j in range(1, n_col + 1):
                if i % 2 == 1:
                    n = j * 2
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2
                else:
                    n = j * 2 - 1
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2

                y = i * self.a * (3 ** 0.5) / 2

                x_coords_right.append(x_right)
                y_coords_right.append(y)
                x_coords_left.append(x_left)
                y_coords_left.append(y)

        return (np.array(x_coords_right), np.array(y_coords_right)), (np.array(x_coords_left), np.array(y_coords_left))

    def _create_wg(self, params: list):
        n_cols_start = len(params)
        a_poly = 7.79169e-05
        n_poly = 2 * n_cols_start
        sf1_s = 1.01386
        sf2_s = 1.09038
        b_poly = (sf1_s - sf2_s) / (1 - n_poly) - a_poly * (1 + n_poly)
        c_poly = sf1_s - a_poly * sf1_s ** 2 - b_poly * sf1_s
        sf_vals = [a_poly * i ** 2 + b_poly * i + c_poly for i in range(1, n_poly + 1)]

        tab_space = np.array(params)
        cumul = np.cumsum(params)
        wg_conn_add_period = 0.411679
        wg_conn_add = self.a * sf1_s * wg_conn_add_period
        x0 = np.round(self.d_fregion + self.a * cumul[-1] - wg_conn_add, 6)
        wg_length = self.membrane_length / 2 - x0
        x_min_coords = [x0, x0, -self.membrane_length / 2, -self.membrane_length / 2]
        y_min_coords = [self.wg1_width / 2, -(self.wg1_width / 2 + self.trench_width), self.wg1_width / 2,
                        -(self.wg1_width / 2 + self.trench_width)]
        x_max_coords = [self.membrane_length / 2, self.membrane_length / 2, -x0, -x0]
        y_max_coords = [self.wg1_width / 2 + self.trench_width, -self.wg1_width / 2,
                        self.wg1_width / 2 + self.trench_width, -self.wg1_width / 2]

        return np.array(x_min_coords), np.array(y_min_coords), np.array(x_max_coords), np.array(y_max_coords)



class PhcForOptimization(GeomBase):
    def __init__(self):
        self.wg_region_width = 1.5e-6
        self.wg_width = 0.6e-6
        self.wg_region_height = 0.173e-6
        self.n_wg = 3.46

        self.membrane_length = 22.3844
        self.membrane_width = 11.8174
        self.trench_width = 1.0
        self.wg_z_span = 0.176376
        self.wg1_width = 0.397017

        self.x_span = 2 * 8.69221e-6
        self.y_span = 8e-6
        self.z_span = 2e-6

        self.ny_fr = 10
        self.hole_radius = 0.077e-6
        self.a = 0.246
        self.d_fregion = 2.46
        self.lambda0 = 0.94e-6
        self.lambda_span = 0.02e-6

        self.dx = 0.03e-6
        self.dy = 0.03e-6
        self.dz = 0.03e-6

        self._source_pos_x = -self.x_span / 2
        self._fom_pos_x = self.x_span / 2

        self._mode_span_y = 1.5e-6
        self._mode_span_z = 0.881e-6

    def setup_script(self, fdtd: FDTD):

        filename = "C:\\Users\\TimothéeMounier\\Desktop\\optimizationconnection WG_PhC\\check geo\\fsf_crystal_3d_sftaper.fsp"

        fdtd.load(filename)

        fdtd.addmode(name="source",
                     injection_axis="x-axis",
                     direction="forward",
                     x=self._source_pos_x,
                     y=0, z=0,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     mode_selection="fundamental TE mode",
                     center_wavelength=self.lambda0,
                     wavelength_span=self.lambda_span)

        fdtd.setglobalsource("center wavelength", self.lambda0)
        fdtd.setglobalsource("wavelength span", self.lambda_span)
        fdtd.setglobalmonitor("frequency points", 100)

        fdtd.addpower(name="fom",
                      monitor_type="2D X-normal",
                      x=self._fom_pos_x,
                      y=0, z=0,
                      y_span=self._mode_span_y,
                      z_span=self._mode_span_z,
                      override_global_monitor_settings=True,
                      frequency_points=100
                      )

        fdtd.addmesh(name="fom_mesh",
                     x=self._fom_pos_x,
                     y=0, z=0,
                     x_span=2 * self.dx,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     override_x_mesh=True,
                     override_y_mesh=False,
                     override_z_mesh=False,
                     dx=self.dx,
                     )

        fdtd.addpower(name="opt_fields",
                      monitor_type="2D Z-normal",
                      x=0, y=0, z=0,
                      x_span=self.x_span,
                      y_span=self._mode_span_y,
                      override_global_monitor_settings=True,
                      frequency_points=20,
                      spatial_interpolation="specified position",
                      record_data_in_pml=True
                      )

        fdtd.addmesh(name="opt_fields_mesh",
                     x=0, y=0, z=0,
                     x_span=self.x_span,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     dx=self.dx, dy=self.dy, dz=self.dz)

        fdtd.redrawon()


    def setup_geometry(self, params: list, fdtd: FDTD):

        (x_coords_right, y_coords_right), (x_coords_left, y_coords_left) = self._create_fast_regions()
        (x_min_coords, y_min_coords, x_max_coords, y_max_coords) = self._create_wg(params)

        #Adding rect for WG
        for i, (x_min, y_min, x_max, y_max) in enumerate(zip(x_min_coords, y_min_coords, x_max_coords, y_max_coords)):
            fdtd.addrect(name=f'rect{i}', x_min=x_min * 1e-6, y_min=y_min * 1e-6, x_max=x_max * 1e-6,
                         y_max=y_max * 1e-6, z=0, z_span=0.176376e-6, index=1.4, material="etch")

        # Adding circles for the right region
        for i, (x, y) in enumerate(zip(x_coords_right, y_coords_right)):
            fdtd.addcircle(name=f'circle_right_{i}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                           radius=self.hole_radius, index=1.4, material="etch")

        # Adding circles for the left region
        for j, (x, y) in enumerate(zip(x_coords_left, y_coords_left)):
            fdtd.addcircle(name=f'circle_left_{j}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                           radius=self.hole_radius, index=1.4, material="etch")

    def update_geometry(self, params: list, fdtd: FDTD):

        (x_coords_right, y_coords_right), (x_coords_left, y_coords_left) = self._create_fast_regions()
        (x_min_coords, y_min_coords, x_max_coords, y_max_coords) = self._create_wg(params)

        for i, (x, y) in enumerate(zip(x_coords_right, y_coords_right)):
            coordinates = {"x": x * 1e-6,
                           "y": y * 1e-6,
                           "z": 0,
                           "z span": 0.176376e-6,
                           }
            fdtd.setnamed(f'circle_right_{i}', coordinates)

        for i, (x, y) in enumerate(zip(x_coords_left, y_coords_left)):
            coordinates = {"x": x * 1e-6,
                           "y": y * 1e-6,
                           "z": 0,
                           "z span": 0.176376e-6,
                           }
            fdtd.setnamed(f'circle_left_{i}', coordinates)

        for i, (x_min, y_min, x_max, y_max) in enumerate(zip(x_min_coords, y_min_coords, x_max_coords, y_max_coords)):
            coordinatesr = {"x min": x_min * 1e-6,
                            "x max": x_max * 1e-6,
                            "y min": y_min * 1e-6,
                            "y max": y_max * 1e-6,
                            "z": 0,
                            "z span": 0.176376e-6,
                            }
            fdtd.setnamed(f'rect{i}', coordinatesr)

    def _create_fast_regions(self, params : list):
        n_cols_start = 11
        sf1_s = params[0]
        sf2_s = params[1]
        a_poly = 7.79169e-05
        n_poly = 2 * n_cols_start
        b_poly = (sf1_s - sf2_s) / (1 - n_poly) - a_poly * (1 + n_poly)
        c_poly = sf1_s - a_poly * sf1_s ** 2 - b_poly * sf1_s
        sf_vals = [a_poly * i ** 2 + b_poly * i + c_poly for i in range(1, n_poly + 1)]
        sf_vals = np.round(sf_vals, 5)
        tab_space = np.array(sf_vals)
        cumul = np.cumsum(tab_space)

        x_coords_right = []
        y_coords_right = []
        x_coords_left = []
        y_coords_left = []

        for i in range(-self.ny_fr, self.ny_fr + 1):
            for j in range(1, n_cols_start + 1):
                if i == 0:
                    continue
                if i % 2 == 1:
                    n = j * 2
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2
                else:
                    n = j * 2 - 1
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2
                if i == 1:
                    y = i * self.a * (3 ** 0.5) / 2
                else:
                    y = i * self.a * (3 ** 0.5) / 2

                x_coords_right.append(x_right)
                y_coords_right.append(y)
                x_coords_left.append(x_left)
                y_coords_left.append(y)

        return (np.array(x_coords_right), np.array(y_coords_right)), (np.array(x_coords_left), np.array(y_coords_left))

    def _create_wg(self, params: list):
        n_cols_start = 11
        a_poly = 7.79169e-05
        n_poly = 2 * n_cols_start
        sf1_s = 1.01386
        sf2_s = 1.09038
        b_poly = (sf1_s - sf2_s) / (1 - n_poly) - a_poly * (1 + n_poly)
        c_poly = sf1_s - a_poly * sf1_s ** 2 - b_poly * sf1_s
        sf_vals = [a_poly * i ** 2 + b_poly * i + c_poly for i in range(1, n_poly + 1)]
        tab_space = np.array(sf_vals)
        cumul = np.cumsum(tab_space)
        wg_conn_add_period = 0.411679
        wg_conn_add = self.a * sf1_s * wg_conn_add_period
        x0 = self.d_fregion + self.a * cumul[2 * n_cols_start - 1] / 2 - wg_conn_add
        wg_length = self.membrane_length / 2 - x0
        x_min_coords = [x0, x0, -self.membrane_length / 2, -self.membrane_length / 2]
        y_min_coords = [self.wg1_width / 2, -(self.wg1_width / 2 + self.trench_width), self.wg1_width / 2,
                        -(self.wg1_width / 2 + self.trench_width)]
        x_max_coords = [self.membrane_length / 2, self.membrane_length / 2, -x0, -x0]
        y_max_coords = [self.wg1_width / 2 + self.trench_width, -self.wg1_width / 2,
                        self.wg1_width / 2 + self.trench_width, -self.wg1_width / 2]

        return np.array(x_min_coords), np.array(y_min_coords), np.array(x_max_coords), np.array(y_max_coords)


class Phc_Paper(GeomBase):
    def __init__(self):
        self.wg_region_width = 1.5e-6
        self.wg_width = 0.6e-6
        self.wg_region_height = 0.173e-6
        self.n_wg = 3.46

        self.membrane_length = 22.3844
        self.membrane_width = 11.8174
        self.trench_width = 1.0
        self.wg_z_span = 0.176376
        self.wg1_width = 0.397017

        self.x_span = 2 * 8.69221e-6
        self.y_span = 8e-6
        self.z_span = 2e-6

        self.ny_fr = 10
        self.hole_radius = 0.077e-6
        self.a = 0.246
        self.d_fregion = 2.46
        self.lambda0 = 0.94e-6
        self.lambda_span = 0.02e-6

        self.dx = 0.03e-6
        self.dy = 0.03e-6
        self.dz = 0.03e-6

        self._source_pos_x = -self.x_span / 2
        self._fom_pos_x = self.x_span / 2

        self._mode_span_y = 1.5e-6
        self._mode_span_z = 0.881e-6

    def setup_script(self, fdtd: FDTD):

        filename = "C:\\Users\\TimothéeMounier\\Documents\\Optimization\\optim FDTD\\Phc_slow_paper_wg.fsp"

        fdtd.load(filename)

        # props = OrderedDict([
        #     ("dimension", "3D"),
        #     ("x", 0.),
        #     ("x span", self.x_span),
        #     ("y", 0.),
        #     ("y span", self.y_span),
        #     ("z", 0.),
        #     ("z span", self.z_span),
        #     ("y min bc", "anti-symmetric"),
        #     ("mesh accuracy", 2),
        #     ("simulation time", 10000e-15),
        #     ("auto shutoff min", 1e-3),
        # ])
        #
        # fdtd.addfdtd(properties=props)

        # fdtd.addrect(name="membrane",
        #              x=0, y=0, z=0,
        #              x_span=self.membrane_length * 1e-6,
        #              y_span=self.membrane_width * 1e-6,
        #              z_span=self.wg_region_height,
        #              index=self.n_wg)
        #
        # fdtd.addrect(name="inner membrane",
        #              x=0, y=0, z=0.0316e-6,
        #              x_span=self.membrane_length * 1e-6,
        #              y_span=self.membrane_width * 1e-6,
        #              z_span=0.047e-6,
        #              index=self.n_wg)

        fdtd.addmode(name="source",
                     injection_axis="x-axis",
                     direction="forward",
                     x=self._source_pos_x,
                     y=0, z=0,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     mode_selection="fundamental TE mode",
                     center_wavelength=self.lambda0,
                     wavelength_span=self.lambda_span)

        fdtd.setglobalsource("center wavelength", self.lambda0)
        fdtd.setglobalsource("wavelength span", self.lambda_span)
        fdtd.setglobalmonitor("frequency points", 100)

        fdtd.addpower(name="fom",
                      monitor_type="2D X-normal",
                      x=self._fom_pos_x,
                      y=0, z=0,
                      y_span=self._mode_span_y,
                      z_span=self._mode_span_z,
                      override_global_monitor_settings=True,
                      frequency_points=100
                      )

        fdtd.addmesh(name="fom_mesh",
                     x=self._fom_pos_x,
                     y=0, z=0,
                     x_span=2 * self.dx,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     override_x_mesh=True,
                     override_y_mesh=False,
                     override_z_mesh=False,
                     dx=self.dx,
                     )

        fdtd.addpower(name="opt_fields",
                      monitor_type="2D Z-normal",
                      x=0, y=0, z=0,
                      x_span=self.x_span,
                      y_span=self._mode_span_y,
                      override_global_monitor_settings=True,
                      frequency_points=20,
                      spatial_interpolation="specified position",
                      record_data_in_pml=True
                      )

        fdtd.addmesh(name="opt_fields_mesh",
                     x=0, y=0, z=0,
                     x_span=self.x_span,
                     y_span=self._mode_span_y,
                     z_span=self._mode_span_z,
                     dx=self.dx, dy=self.dy, dz=self.dz)

        fdtd.redrawon()


    def setup_geometry(self, params: list, fdtd: FDTD):

        (x_coords_right, y_coords_right), (x_coords_left, y_coords_left) = self._create_fast_regions_paper_geometry()
        # (x_min_coords, y_min_coords, x_max_coords, y_max_coords) = self._create_wg(params)

        # Adding rect for WG
        # for i, (x_min, y_min, x_max, y_max) in enumerate(zip(x_min_coords, y_min_coords, x_max_coords, y_max_coords)):
        #     fdtd.addrect(name=f'rect{i}', x_min=x_min * 1e-6, y_min=y_min * 1e-6, x_max=x_max * 1e-6,
        #                  y_max=y_max * 1e-6, z=0, z_span=0.176376e-6, index=1.4, material="etch")

        # Adding circles for the right region
        for i, (x, y) in enumerate(zip(x_coords_right, y_coords_right)):
            fdtd.addcircle(name=f'circle_right_{i}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                           radius=self.hole_radius, index=1.4, material="etch")

        # Adding circles for the left region
        for j, (x, y) in enumerate(zip(x_coords_left, y_coords_left)):
            fdtd.addcircle(name=f'circle_left_{j}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                           radius=self.hole_radius, index=1.4, material="etch")

    def update_geometry(self, params: list, fdtd: FDTD):

        (x_coords_right, y_coords_right), (x_coords_left, y_coords_left) = self._create_fast_regions_paper_geometry()
        # (x_min_coords, y_min_coords, x_max_coords, y_max_coords) = self._create_wg(params)

        for i, (x, y) in enumerate(zip(x_coords_right, y_coords_right)):
            coordinates = {"x": x * 1e-6,
                           "y": y * 1e-6,
                           "z": 0,
                           "z span": 0.176376e-6,
                           }
            fdtd.setnamed(f'circle_right_{i}', coordinates)

        for i, (x, y) in enumerate(zip(x_coords_left, y_coords_left)):
            coordinates = {"x": x * 1e-6,
                           "y": y * 1e-6,
                           "z": 0,
                           "z span": 0.176376e-6,
                           }
            fdtd.setnamed(f'circle_left_{i}', coordinates)

        # for i, (x_min, y_min, x_max, y_max) in enumerate(zip(x_min_coords, y_min_coords, x_max_coords, y_max_coords)):
        #     coordinatesr = {"x min": x_min * 1e-6,
        #                     "x max": x_max * 1e-6,
        #                     "y min": y_min * 1e-6,
        #                     "y max": y_max * 1e-6,
        #                     "z": 0,
        #                     "z span": 0.176376e-6,
        #                     }
        #     fdtd.setnamed(f'rect{i}', coordinatesr)

    def _create_fast_regions_false_geom(self):
        n_col = 11
        sf1 = 1.01383
        sf2 = 1.09038

        stop_air = 5
        n_colfin = n_col + stop_air // 2
        a_poly = 7.79169e-05
        n_poly = 2 * n_colfin
        b_poly = (sf1 - sf2) / (1 - n_poly) - a_poly * (1 + n_poly)
        c_poly = sf1 - a_poly * sf1 ** 2 - b_poly * sf1
        sf_vals = [a_poly * i ** 2 + b_poly * i + c_poly for i in range(1, n_poly + 1)]

        tab_space = np.array(sf_vals)
        cumul = np.cumsum(tab_space)

        x_coords_right = []
        y_coords_right = []
        x_coords_left = []
        y_coords_left = []
        for i in range(-self.ny_fr, self.ny_fr + 1):
            # elif abs(i) < stop_air and i % 2 == 0:
            #     n_cols = n_col - shift + abs(i) // 2
            # elif abs(i) < stop_air and i % 2 == 1:
            #     n_cols = n_col - shift + abs(i) // 2
            # else:
            #     n_cols = n_col
            if i == 0:
                continue
            elif abs(i) < stop_air:
                n_cols = n_col + abs(i) // 2
            else:
                n_cols = n_colfin
            for j in range(1, n_cols + 1):
                if i % 2 == 1:
                    n = j * 2
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2
                else:

                    n = j * 2 - 1
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2

                y = i * self.a * (3 ** 0.5) / 2

                x_coords_right.append(x_right)
                y_coords_right.append(y)
                x_coords_left.append(x_left)
                y_coords_left.append(y)

        return (np.array(x_coords_right), np.array(y_coords_right)), (np.array(x_coords_left), np.array(y_coords_left))

    def _create_fast_regions_paper_geometry(self):
        n_cols_start = 11
        sf1_s = 1.01383
        sf2_s = 1.09038
        sf1_t = 1.01383
        sf2_t = 1.11038
        stop_air = 6
        n_colfin = n_cols_start + stop_air // 2

        a_poly = 7.79169e-05
        n_poly = 2 * n_cols_start
        b_poly = (sf1_s - sf2_s) / (1 - n_poly) - a_poly * (1 + n_poly)
        c_poly = sf1_s - a_poly * sf1_s ** 2 - b_poly * sf1_s
        sf_vals = [a_poly * i ** 2 + b_poly * i + c_poly for i in range(1, n_poly + 1)]

        tab_space = np.array(sf_vals)
        cumul = np.cumsum(tab_space)

        a_poly_2 = 7.79169e-05
        n_poly_2 = 2 * n_colfin
        b_poly_2 = (sf1_t - sf2_t) / (1 - n_poly_2) - a_poly_2 * (1 + n_poly_2)
        c_poly_2 = sf1_t - a_poly_2 * sf1_t ** 2 - b_poly_2 * sf1_t
        sf_vals2 = [a_poly_2 * i ** 2 + b_poly_2 * i + c_poly_2 for i in range(1, n_poly_2 + 1)]
        tab_space2 = np.array(sf_vals2)
        cumul2 = np.cumsum(tab_space2)

        x_coords_right = []
        y_coords_right = []
        x_coords_left = []
        y_coords_left = []

        for i in range(-self.ny_fr, self.ny_fr + 1):
            for j in range(1, n_cols_start + 1):
                if i == 0:
                    continue
                if i % 2 == 1:
                    n = j * 2
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2
                else:
                    n = j * 2 - 1
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2

                y = i * self.a * (3 ** 0.5) / 2

                x_coords_right.append(x_right)
                y_coords_right.append(y)
                x_coords_left.append(x_left)
                y_coords_left.append(y)

        for i in range(-self.ny_fr, self.ny_fr + 1):
            if i == 0:
                continue
            elif abs(i) < stop_air:
                n_cols2 = n_cols_start + 1 + abs(i) // 2
            else:
                n_cols2 = n_colfin
            for j in range(n_cols_start + 1, n_cols2):
                if i % 2 == 1:
                    n = j * 2
                    x_right = self.d_fregion + self.a * cumul2[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul2[n - 1] / 2
                else:
                    n = j * 2 - 1
                    x_right = self.d_fregion + self.a * cumul2[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul2[n - 1] / 2

                y = i * self.a * (3 ** 0.5) / 2

                x_coords_right.append(x_right)
                y_coords_right.append(y)
                x_coords_left.append(x_left)
                y_coords_left.append(y)

        return (np.array(x_coords_right), np.array(y_coords_right)), (np.array(x_coords_left), np.array(y_coords_left))

    def _create_fast_regions_(self):
        n_cols_start = 11
        sf1_s = 1.01386
        sf2_s = 1.09038
        sf1_t = 1.20383
        sf2_t = 1.30038

        stop_air = 6
        n_colfin = n_cols_start + stop_air

        a_poly = 7.79169e-05
        n_poly = 2 * n_cols_start
        b_poly = (sf1_s - sf2_s) / (1 - n_poly) - a_poly * (1 + n_poly)
        c_poly = sf1_s - a_poly * sf1_s ** 2 - b_poly * sf1_s
        sf_vals = [a_poly * i ** 2 + b_poly * i + c_poly for i in range(1, n_poly + 1)]
        sf_vals = np.round(sf_vals, 5)
        tab_space = np.array(sf_vals)
        cumul = np.cumsum(tab_space)

        x_coords_right = []
        y_coords_right = []
        x_coords_left = []
        y_coords_left = []

        for i in range(-self.ny_fr, self.ny_fr + 1):
            for j in range(1, n_cols_start + 1):
                if i == 0:
                    continue
                if i % 2 == 1:
                    n = j * 2
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2
                else:
                    n = j * 2 - 1
                    x_right = self.d_fregion + self.a * cumul[n - 1] / 2
                    x_left = -self.d_fregion - self.a * cumul[n - 1] / 2
                if i == 1:
                    y = i * self.a * (3 ** 0.5) / 2
                else:
                    y = i * self.a * (3 ** 0.5) / 2

                x_coords_right.append(x_right)
                y_coords_right.append(y)
                x_coords_left.append(x_left)
                y_coords_left.append(y)

        # for i in range(-ny_fr, ny_fr + 1):
        #     if abs(i) == 1:
        #         n_cols2 = n_cols_start
        #     elif abs(i) < stop_air:
        #         n_cols2 = n_cols_start +abs(i) //2
        #     else:
        #         n_cols2 = n_colfin
        #     for j in range(n_cols_start + 1, n_cols2):
        #         if i % 2 == 1:
        #             n = j * 2
        #             print(cumul2[n-1])
        #             x_right = d_fregion + a * cumul2[n -1] / 2
        #             x_left = -d_fregion - a * cumul2[n -1] / 2
        #         else:
        #             n = j * 2 - 1
        #             x_right = d_fregion + a * cumul2[n -1] / 2
        #             x_left = -d_fregion - a * cumul2[n- 1] / 2
        #         y = i * a*(3 ** 0.5) / 2
        #         x_coords_right.append(x_right)
        #         y_coords_right.append(y)
        #         x_coords_left.append(x_left)
        #         y_coords_left.append(y)

        return (np.array(x_coords_right), np.array(y_coords_right)), (np.array(x_coords_left), np.array(y_coords_left))

    def _create_wg(self, params: list):
        n_cols_start = 11
        a_poly = 7.79169e-05
        n_poly = 2 * n_cols_start
        sf1_s = 1.01386
        sf2_s = 1.09038
        b_poly = (sf1_s - sf2_s) / (1 - n_poly) - a_poly * (1 + n_poly)
        c_poly = sf1_s - a_poly * sf1_s ** 2 - b_poly * sf1_s
        sf_vals = [a_poly * i ** 2 + b_poly * i + c_poly for i in range(1, n_poly + 1)]

        tab_space = np.array(sf_vals)
        cumul = np.cumsum(tab_space)
        wg_conn_add_period = 0.411679
        wg_conn_add = self.a * sf1_s * wg_conn_add_period
        x0 = self.d_fregion + self.a * cumul[2 * n_cols_start - 1] / 2 - wg_conn_add
        wg_length = self.membrane_length / 2 - x0
        x_min_coords = [x0, x0, -self.membrane_length / 2, -self.membrane_length / 2]
        y_min_coords = [self.wg1_width / 2, -(self.wg1_width / 2 + self.trench_width), self.wg1_width / 2,
                        -(self.wg1_width / 2 + self.trench_width)]
        x_max_coords = [self.membrane_length / 2, self.membrane_length / 2, -x0, -x0]
        y_max_coords = [self.wg1_width / 2 + self.trench_width, -self.wg1_width / 2,
                        self.wg1_width / 2 + self.trench_width, -self.wg1_width / 2]

        return np.array(x_min_coords), np.array(y_min_coords), np.array(x_max_coords), np.array(y_max_coords)

