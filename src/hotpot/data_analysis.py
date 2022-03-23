import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .functools import FuncArray
from .geometry.primiary import Cartesian3, Segment


grid_center_x, grid_center_y = np.meshgrid(np.linspace(-24.5,24.5,50), np.linspace(-24.5,24.5,50))

grid_border_lower_x, grid_border_lower_y = np.meshgrid(np.linspace(-25,24,50), np.linspace(-25,24,50))
grid_border_upper_x, grid_border_upper_y = np.meshgrid(np.linspace(-24,25,50), np.linspace(-24,25,50))

grid_center = np.stack([grid_center_x, grid_center_y], axis=-1)
grid_border = np.stack([grid_border_upper_x, grid_border_upper_y, grid_border_lower_x, grid_border_lower_y], axis=-1)


Cartesian3_xy_mag = lambda c3: np.mean(np.sqrt(np.square(c3.x) + np.square(c3.y)))
Cartesian3_xyz_distance_nrom = lambda c3: np.mean(np.sqrt(np.square(c3.x) + np.square(c3.y) + np.square(c3.z)))

r50_r90_mean_of = lambda arr: (
    np.percentile(arr, 50),
    np.percentile(arr, 90),
    np.mean(arr)
)

Cartesian3_xy_mean = lambda c3: np.array([Cartesian3_xy_mag(c3), np.mean(c3.x), np.mean(c3.y)])


def map_by_griding_mask(data, func: "do sth with indexed data", mask):
    tmp = []
    for x in range(mask.shape[0]):
        tmp_inner = []
        for y in range(mask.shape[0]):
            tmp_inner.append(func(data[mask[x,y]]))
        tmp.append(tmp_inner)
    return tmp


def plot_xy_bias_vector(xy_error, bias_vector, xy_origins):
    fig = plt.figure(1,figsize=(8, 8))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        xy_error,
        cmap='viridis',
        aspect='equal',
        extent=[-25,25,25,-25],
        vmin=np.array(xy_error).min(), 
        vmax=np.array(xy_error).max()
    )


    for i in range(50):
        for j in range(50):
            ax.annotate(
                "",
                xy=bias_vector[i,j,:].tolist(), 
                xycoords='data',
                xytext=xy_origins[i,j,:].tolist(), 
                textcoords='data',
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3"
                )
            )

    fig.colorbar(im, ax=ax)
    plt.savefig('tmp.png', dpi=500)


class CachedSysetemData:
    def __init__(self, cached_csv_path):
        self.raw_sample = pd.read_csv(cached_csv_path)
        self.source = Cartesian3.from_xyz(
                self.raw_sample.sourcePosX,
                self.raw_sample.sourcePosY,
            self.raw_sample.sourcePosZ,
        )
        self.real_lor_global = Segment(
            Cartesian3.from_xyz(
                self.raw_sample.gamma_1_x,
                self.raw_sample.gamma_1_y,
                self.raw_sample.gamma_1_z,
            ),
            Cartesian3.from_xyz(
                self.raw_sample.gamma_2_x,
                self.raw_sample.gamma_2_y,
                self.raw_sample.gamma_2_z,
            )
        )
        
        self.real_lor_local = Segment(
            Cartesian3.from_xyz(
                self.raw_sample.gamma_1_local_x,
                self.raw_sample.gamma_1_local_y,
                self.raw_sample.gamma_1_local_z,
            ),
            Cartesian3.from_xyz(
                self.raw_sample.gamma_2_local_x,
                self.raw_sample.gamma_2_local_y,
                self.raw_sample.gamma_2_local_z,
            )
        )

        self.anger_infered_lor_global = Segment(
            Cartesian3.from_xyz(
                self.raw_sample.anger_infered_gamma_1_global_pos_x,
                self.raw_sample.anger_infered_gamma_1_global_pos_y,
                self.raw_sample.anger_infered_gamma_1_global_pos_z,
            ),
            Cartesian3.from_xyz(
                self.raw_sample.anger_infered_gamma_2_global_pos_x,
                self.raw_sample.anger_infered_gamma_2_global_pos_y,
                self.raw_sample.anger_infered_gamma_2_global_pos_z,
            )
        )

        self.anger_infered_lor_local = Segment(
            Cartesian3.from_xyz(
                self.raw_sample.anger_gamma_1_local_pos_x,
                self.raw_sample.anger_gamma_1_local_pos_y,
                self.raw_sample.anger_gamma_1_local_pos_z,
            ),
            Cartesian3.from_xyz(
                self.raw_sample.anger_gamma_2_local_pos_x,
                self.raw_sample.anger_gamma_2_local_pos_y,
                self.raw_sample.anger_gamma_2_local_pos_z,
            )
        )

        self.net_infered_lor_global = Segment(
            Cartesian3.from_xyz(
                self.raw_sample.net_infered_gamma_1_global_pos_x,
                self.raw_sample.net_infered_gamma_1_global_pos_y,
                self.raw_sample.net_infered_gamma_1_global_pos_z,
            ),
            Cartesian3.from_xyz(
                self.raw_sample.net_infered_gamma_2_global_pos_x,
                self.raw_sample.net_infered_gamma_2_global_pos_y,
                self.raw_sample.net_infered_gamma_2_global_pos_z,
            )
        )

        self.net_infered_lor_local = Segment(
            Cartesian3.from_xyz(
                self.raw_sample.net_infered_gamma_1_local_pos_x,
                self.raw_sample.net_infered_gamma_1_local_pos_y,
                self.raw_sample.net_infered_gamma_1_local_pos_z,
            ),
            Cartesian3.from_xyz(
                self.raw_sample.net_infered_gamma_2_local_pos_x,
                self.raw_sample.net_infered_gamma_2_local_pos_y,
                self.raw_sample.net_infered_gamma_2_local_pos_z,
            )
        )

        self.single_crystal_net_infered_lor_local = Segment(
            Cartesian3.from_xyz(
                self.raw_sample.single_crystal_net_infered_gamma_1_local_x,
                self.raw_sample.single_crystal_net_infered_gamma_1_local_y,
                self.raw_sample.single_crystal_net_infered_gamma_1_local_z,
            ),
            Cartesian3.from_xyz(
                self.raw_sample.single_crystal_net_infered_gamma_2_local_x,
                self.raw_sample.single_crystal_net_infered_gamma_2_local_y,
                self.raw_sample.single_crystal_net_infered_gamma_2_local_z,
            )
        )

        self.single_crystal_net_infered_lor_global = Segment(
            Cartesian3.from_xyz(
                self.raw_sample.single_crystal_net_infered_gamma_1_global_x,
                self.raw_sample.single_crystal_net_infered_gamma_1_global_y,
                self.raw_sample.single_crystal_net_infered_gamma_1_global_z,
            ),
            Cartesian3.from_xyz(
                self.raw_sample.single_crystal_net_infered_gamma_2_global_x,
                self.raw_sample.single_crystal_net_infered_gamma_2_global_y,
                self.raw_sample.single_crystal_net_infered_gamma_2_global_z,
            )
        )


    def __getitem__(self, idx):
        return self.raw_sample.iloc[idx]
       
    @property
    def griding_mask(self):
        return self.real_lor_local.hstack().griding_by(grid_border)


    @property
    def net_xy_error_with_bias_vector(self):
        return np.asarray(map_by_griding_mask(
            self.net_infered_lor_local.hstack() - self.real_lor_local.hstack(),
            Cartesian3_xy_mean,
            self.griding_mask
        ))

    def plot_net_xy_bias_vector(self):
        plot_xy_bias_vector(
            self.net_xy_error_with_bias_vector[:,:,0], 
            self.net_xy_error_with_bias_vector[:,:,1:] + grid_center, 
            grid_center
        ) 

    @property
    def anger_xy_error_with_bias_vector(self):
        return np.asarray(map_by_griding_mask(
            self.anger_infered_lor_local.hstack() - self.real_lor_local.hstack(),
            Cartesian3_xy_mean,
            self.griding_mask
        ))

    def plot_anger_xy_bias_vector(self):
        plot_xy_bias_vector(
            self.anger_xy_error_with_bias_vector[:,:,0], 
            self.anger_xy_error_with_bias_vector[:,:,1:] + grid_center, 
            grid_center
        )

    @property
    def single_crystal_net_xy_error_with_bias_vector(self):
        return np.asarray(map_by_griding_mask(
            self.single_crystal_net_infered_lor_local.hstack() - self.real_lor_local.hstack(),
            Cartesian3_xy_mean,
            self.griding_mask
        ))

    def plot_single_crystal_net_xy_bias_vector(self):
        plot_xy_bias_vector(
            self.single_crystal_net_xy_error_with_bias_vector[:,:,0],
            self.single_crystal_net_xy_error_with_bias_vector[:,:,1:] + grid_center,
            grid_center
        )

    @property
    def net_mean_mae(self):
        return np.array(
            map_by_griding_mask(
                self.net_infered_lor_local.hstack() - self.real_lor_local.hstack(),
                lambda c_3: np.mean(c_3.length_as_vector()),
                self.griding_mask
            )       
        )

    def plot_net_mean_mae(self):
        fig = plt.figure(1,figsize=(10, 10))
        ax = fig.add_subplot(111)

        im = ax.imshow(
            self.net_mean_mae,
            cmap='viridis',
            aspect='equal',
            extent=[-25,25,25,-25],
            vmin=np.array(self.net_mean_mae).min(),
            vmax=np.array(self.net_mean_mae).max()
        )
        fig.colorbar(im, ax=ax)
        plt.savefig('tmp.png', dpi=300)


    @property
    def anger_mean_mae(self):
        return np.array(
            map_by_griding_mask(
                self.anger_infered_lor_local.hstack() - self.real_lor_local.hstack(),
                lambda c_3: np.mean(c_3.length_as_vector()),
                self.griding_mask
            )       
        )

    def plot_anger_mean_mae(self):
        fig = plt.figure(1,figsize=(10, 10))
        ax = fig.add_subplot(111)

        im = ax.imshow(
            self.anger_mean_mae,
            cmap='viridis',
            aspect='equal',
            extent=[-25,25,25,-25],
            vmin=np.array(self.anger_mean_mae).min(),
            vmax=np.array(self.anger_mean_mae).max()
        )
        fig.colorbar(im, ax=ax)
        plt.savefig('tmp.png', dpi=300)

    @property
    def single_crystal_net_mae(self):
        return np.array(
            map_by_griding_mask(
                self.single_crystal_net_infered_lor_local.hstack() - self.real_lor_local.hstack(),
                lambda c_3: np.mean(c_3.length_as_vector()),
                self.griding_mask
            )
        )

    def plot_single_crystal_net_mae(self):
        fig = plt.figure(1,figsize=(10, 10))
        ax = fig.add_subplot(111)

        im = ax.imshow(
            self.single_crystal_net_mae,
            cmap='viridis',
            aspect='equal',
            extent=[-25,25,25,-25],
            vmin=np.array(self.single_crystal_net_mae).min(),
            vmax=np.array(self.single_crystal_net_mae).max()
        )
        fig.colorbar(im, ax=ax)
        plt.savefig('tmp.png', dpi=300)
