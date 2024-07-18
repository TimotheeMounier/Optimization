import numpy as np
from abc import ABC, abstractmethod
from collections import OrderedDict

from lumapi import MODE

class GeomBase(ABC):
    @abstractmethod
    def setup_script(self, mode: MODE):
        """ Sets up the base simulation """
        pass

    @abstractmethod
    def setup_geometry(self, params: list,mode: MODE ):
        """ Sets up the optimizable geometry """
        pass

    @abstractmethod
    def update_geometry(self, params: list, mode: MODE):
        """ Updates the optimizable geometry """
        pass

    def check_setup(self, params: list, filename=None):
        if filename is None:
            filename = f"{self.__class__.__name__}_geometry.fsp"

        mode = MODE()
        self.setup_script(mode)
        self.setup_geometry(params, mode)
        mode.save(filename)

        return mode


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

        self.x_span = 15e-6
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

        self._source_pos_x = -self.x_span / 2 + 0.6e-6
        self._fom_pos_x = self.x_span / 2 - 0.6e-6

        self._mode_span_y = self.wg_region_width
        self._mode_span_z = self.wg_region_height * 4

    def setup_script(self, mode: MODE):

        filename = "C:\\Users\\TimothéeMounier\\Desktop\\optimizationconnection WG_PhC\\varFDTD\\fsf_taper_3d_var.lms"

        mode.load(filename)

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
        # mode.addvarfdtd(properties=props)

        # mode.addmode(name="source",
        #              injection_axis="x-axis",
        #              direction="forward",
        #              x=self._source_pos_x,
        #              y=0, z=0,
        #              y_span=self._mode_span_y,
        #              z_span=self._mode_span_z,
        #              mode_selection="fundamental TE mode",
        #              center_wavelength=self.lambda0,
        #              wavelength_span=self.lambda_span)
        #
        # mode.setglobalsource("center wavelength", self.lambda0)
        # mode.setglobalsource("wavelength span", self.lambda_span)
        # mode.setglobalmonitor("frequency points", 25)

        # mode.addpower(name="fom",
        #               monitor_type="2D X-normal",
        #               x=self._fom_pos_x,
        #               y=0, z=0,
        #               y_span=self._mode_span_y,
        #               z_span=self._mode_span_z)
        #
        # mode.addmesh(name="fom_mesh",
        #              x=self._fom_pos_x,
        #              y=0, z=0,
        #              x_span=2 * self.dx,
        #              y_span=self._mode_span_y,
        #              z_span=self._mode_span_z,
        #              override_x_mesh=True,
        #              override_y_mesh=False,
        #              override_z_mesh=False,
        #              dx=self.dx)
        #
        # mode.addpower(name="opt_fields",
        #               monitor_type="3D",
        #               x=0, y=0, z=0,
        #               x_span=self.x_span,
        #               y_span=self.wg_region_width,
        #               z_span=self.wg_region_height,
        #               output_Hx=False, output_Hy=False,
        #               output_Hz=False, output_power=False)
        #
        # mode.addmesh(name="opt_fields_mesh",
        #              x=0, y=0, z=0,
        #              x_span=self.x_span,
        #              y_span=self.wg_region_width,
        #              z_span=self.wg_region_height,
        #              dx=self.dx, dy=self.dy, dz=self.dz)

        mode.redrawon()

    def setup_geometry(self, params: list, mode: MODE):

        (x_coords_right, y_coords_right), (x_coords_left, y_coords_left) = self._create_fast_regions()
        (x_min_coords, y_min_coords, x_max_coords, y_max_coords) = self._create_wg(params)

        # Adding rect for WG
        for i, (x_min, y_min, x_max, y_max) in enumerate(zip(x_min_coords, y_min_coords, x_max_coords, y_max_coords)):
            mode.addrect(name=f'rect{i}', x_min=x_min * 1e-6, y_min=y_min * 1e-6, x_max=x_max * 1e-6,
                         y_max=y_max * 1e-6, z=0, z_span=0.176376e-6, index=1.4, material="etch")

        # Adding circles for the right region
        for i, (x, y) in enumerate(zip(x_coords_right, y_coords_right)):
            if (i + 1) % 11 == 0:
                mode.addcircle(name=f'circle_right_{i}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                               radius=self.hole_radius, index=1.4, material="etch")
            else:
                mode.addcircle(name=f'circle_right_{i}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                               radius=self.hole_radius, index=1.4, material="etch")

        # Adding circles for the left region
        for j, (x, y) in enumerate(zip(x_coords_left, y_coords_left)):
            if (j + 1) % 11 == 0:
                mode.addcircle(name=f'circle_left_{j}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                               radius=self.hole_radius, index=1.4, material="etch")
            else:
                mode.addcircle(name=f'circle_left_{j}', x=x * 1e-6, y=y * 1e-6, z=0.0e-9, z_span=0.176376e-6,
                               radius=self.hole_radius, index=1.4, material="etch")

    def update_geometry(self, params: list, mode: MODE):

        (x_coords_right, y_coords_right), (x_coords_left, y_coords_left) = self._create_fast_regions()
        (x_min_coords, y_min_coords, x_max_coords, y_max_coords) = self._create_wg(params)

        for i, (x, y) in enumerate(zip(x_coords_right, y_coords_right)):
            if (i + 1) % 11 == 0:
                coordinates = {"x": x * 1e-6,
                               "y": y * 1e-6,
                               "z": 0,
                               "z span": 0.176376e-6,
                               "radius": self.hole_radius}
                mode.setnamed(f'circle_right_{i}', coordinates)
            else:
                coordinates = {"x": x * 1e-6,
                               "y": y * 1e-6,
                               "z": 0,
                               "z span": 0.176376e-6,
                               }
                mode.setnamed(f'circle_right_{i}', coordinates)

        for i, (x, y) in enumerate(zip(x_coords_left, y_coords_left)):
            if (i + 1) % 11 == 0:
                coordinates = {"x": x * 1e-6,
                               "y": y * 1e-6,
                               "z": 0,
                               "z span": 0.176376e-6,
                               "radius": self.hole_radius
                               }
                mode.setnamed(f'circle_left_{i}', coordinates)
            else:
                coordinates = {"x": x * 1e-6,
                               "y": y * 1e-6,
                               "z": 0,
                               "z span": 0.176376e-6,
                               }
                mode.setnamed(f'circle_left_{i}', coordinates)

        for i, (x_min, y_min, x_max, y_max) in enumerate(zip(x_min_coords, y_min_coords, x_max_coords, y_max_coords)):
            coordinatesr = {"x min": x_min * 1e-6,
                            "x max": x_max * 1e-6,
                            "y min": y_min * 1e-6,
                            "y max": y_max * 1e-6,
                            "z": 0,
                            "z span": 0.176376e-6,
                            }
            mode.setnamed(f'rect{i}', coordinatesr)

    def _create_fast_regions(self):
        n_cols_start = 11
        sf1_s = 1.01386
        sf2_s = 1.09038
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