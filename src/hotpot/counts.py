from .geometry.system import (
    SipmArray,
    Hit,
    FuncDataFrame,
    HitsEventIDMapping,
)

from hotpot.geometry.primiary import (
    Cartesian3, 
    Box,
    Plane,
    Segment, 
    Database, 
    Surface, 
    Trapezoid
)

from .simulation.image_system import ImageSystem
from .functools import FuncArray
from .simulation.sample import SampleWithAnger
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import os


image_system = ImageSystem.from_file(os.getenv("IMAGE_SYSTEM"))
p2g = Database().read_sql(os.getenv("P2G_SQL"))
sipm_array = SipmArray(int(os.getenv("CRYSTAL_Z")))
counts_scale_value = int(os.getenv("COUNTS_SCALE_VALUE"))
crystal_z = int(os.getenv("CRYSTAL_Z"))

pd_df_2_func_array = lambda pd_df: FuncArray(np.array(pd_df.to_numpy().tolist()))
pd_df_2_Cartesian3 = lambda pd_df, keys: Cartesian3.from_tuple3s(pd_df[keys].to_numpy())

def move_z():
    if crystal_z == 15:
        return 217.5
    elif crystal_z == 20:
        return 220
    elif crystal_z == 25:
        return 222.5

def move_arg_by_crystalID(crystalID):
    move_arg = FuncArray(p2g[["move_x", "move_y", "move_z"]].to_numpy())
    move_arg.replace_col_with_constant(2, move_z())

    rotate_arg = FuncArray(
        p2g[["rotate_angle_x", "rotate_angle_y", "rotate_angle_z"]].to_numpy()
    )

    return [move_arg.to_list()[crystalID], rotate_arg.to_list()[crystalID]]
reverse_move_arg_by_crystalID = lambda crystalID: (np.array(move_arg_by_crystalID(crystalID))*-1).tolist()


def sipm_local_to_global(counts, crystalID):

    sipm_global = sipm_array.local_pos
    sipm_global.z = sipm_global.z + counts.flatten() / counts_scale_value

    return sipm_global.move(move_arg_by_crystalID(crystalID)[0]).rotate_ypr(
        move_arg_by_crystalID(crystalID)[1]
    )


class Counts:
    def __init__(self, raw_sample):
        self.crystalIDs = pd_df_2_func_array(raw_sample.crystalID)

        self.counts = pd_df_2_func_array(raw_sample.counts).transpose(axes=(0, 1, 3, 2))
        self.counts_plotly_global = FuncArray(
            [
                (
                    sipm_local_to_global(
                        self.counts[i, 0, :, :].array, self.crystalIDs[i][0].array
                    ),
                    sipm_local_to_global(
                        self.counts[i, 1, :, :].array, self.crystalIDs[i][1].array
                    ),
                )
                for i in range(self.counts.shape[0])
            ]
        )

        self.sourcePos = pd_df_2_Cartesian3(
            raw_sample, ["sourcePosX", "sourcePosY", "sourcePosZ"]
        )

        self.gamma_1_global = pd_df_2_Cartesian3(
            raw_sample, ["gamma_1_x", "gamma_1_y", "gamma_1_z"]
        )
        self.gamma_1_local = pd_df_2_Cartesian3(
            raw_sample, ["gamma_1_local_x", "gamma_1_local_y", "gamma_1_local_z"]
        )

        self.gamma_2_global = pd_df_2_Cartesian3(
            raw_sample, ["gamma_2_x", "gamma_2_y", "gamma_2_z"]
        )
        self.gamma_2_local = pd_df_2_Cartesian3(
            raw_sample, ["gamma_2_local_x", "gamma_2_local_y", "gamma_2_local_z"]
        )

        self.real_lor = Segment(self.gamma_1_global, self.gamma_2_global)

        self.swa = SampleWithAnger(raw_sample)
        self.anger_infered_lor = Segment.from_listmode(
            tf.cast(
                FuncArray(self.swa.gamma_1_anger_global)
                .concatenate_with(FuncArray(self.swa.gamma_2_anger_global), axis=1)
                .to_tensor(),
                dtype=tf.float32,
            ).numpy()
        )

        self.sipm_counts_n_position = (
            self.counts.transpose([0,2,3,1])[:,:,:,0].expand_dims(3)
            .concatenate_with(self.swa.sipm_center_pos[:,0,:,:,:].transpose([0,2,3,1]), axis=3)
            .concatenate_with(self.counts.transpose([0,2,3,1])[:,:,:,1].expand_dims(3), axis=3)
            .concatenate_with(self.swa.sipm_center_pos[:,1,:,:,:].transpose([0,2,3,1]), axis=3)
        )

    def lor_angle_to(self, lines: Segment):
        def get_detector_front_plane(move_arg, rotate_arg): 
            reference_point = (
                Box
                .from_size(*image_system.crystal_size)
                .vertices[[0,1,2]]
                .move([0,0,-image_system.crystal_size_cart3.z])
                .move(move_arg)
                .rotate_ypr(rotate_arg)
            )
            return Plane(reference_point)

        plane_of_gamma = lambda gamma_idx: (
            self.crystalIDs
            .map(lambda crystalIDs: crystalIDs[gamma_idx])
            .map(lambda crystalID: move_arg_by_crystalID(crystalID))
            .map(lambda move_args: get_detector_front_plane(*move_args))
        )
        
        def lor_plane_angle_of_gamma(gamma_idx):
            result = np.array([
                plane.norm_vector.angle_ang(lor.snd-lor.fst)
                for plane, lor in zip(plane_of_gamma(gamma_idx), lines)
            ]).flatten()

            if np.sum(result>90) == len(result):
                return 180 - result
            else:
                return result

        return np.stack([lor_plane_angle_of_gamma(0), lor_plane_angle_of_gamma(1)], axis=1)

    def get_gamma_1_boundary_index(self, distance_to_boundary):
        boundary_high = 25-distance_to_boundary
        boundary_low = -25+distance_to_boundary
        
        idx = np.logical_or(
            np.logical_or(self.gamma_1_local.x>boundary_high, self.gamma_1_local.x<boundary_low),
            np.logical_or(self.gamma_1_local.y>boundary_high, self.gamma_1_local.y<boundary_low)
        )
        return idx

    def get_gamma_2_boundary_index(self, distance_to_boundary):
        boundary_high = 25-distance_to_boundary
        boundary_low = -25+distance_to_boundary
        
        idx = np.logical_or(
            np.logical_or(self.gamma_2_local.x>boundary_high, self.gamma_2_local.x<boundary_low),
            np.logical_or(self.gamma_2_local.y>boundary_high, self.gamma_2_local.y<boundary_low)
        )
        return idx

    def to_plotly(self, idx):
        return [
            # *self.counts_plotly_global[idx]
            # .map(lambda x: x[0].to_plotly_as_mesh3d(opacity=0.8, color="pink", name="optical photon counts"))
            # .array,
            # *self.counts_plotly_global[idx]
            # .map(lambda x: x[1].to_plotly_as_mesh3d(opacity=0.8, color="pink", name="optical photon counts"))
            # .array,
            self.sourcePos[idx].to_plotly(marker=dict(color="red", size=4), name="source position"),
            *self.real_lor[idx].to_plotly_segment(marker=dict(color="gold", size=3), name="real LOR"),
            *self.anger_infered_lor[idx].to_plotly_segment(
                marker=dict(color="blue", size=3), name="anger logic infered LOR"
            ),
            *image_system.to_plotly(),
        ]
