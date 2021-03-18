from .geometry.system import (
    SipmArray,
    Hit,
    FuncDataFrame,
    HitsEventIDMapping,
)

from hotpot.geometry.primiary import Cartesian3, Segment, Database, Surface, Trapezoid

from .simulation.image_system import ImageSystem
from .functools import FuncArray
from .simulation.sample import SampleWithAnger
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import os


image_system = ImageSystem.from_file(os.getenv("IMAGE_SYSTEM"))
# p2g = Database().read_sql(os.getenv("P2G_SQL"))
p2g = pd.read_csv('/home/zengqinglong/jupyters/monolithicCrystal/data/p2g.csv') 
sipm_array = SipmArray(int(os.getenv("CRYSTAL_Z")))
counts_scale_value = int(os.getenv("COUNTS_SCALE_VALUE"))


pd_df_2_func_array = lambda pd_df: FuncArray(np.array(pd_df.to_numpy().tolist()))
pd_df_2_Cartesian3 = lambda pd_df, keys: Cartesian3.from_tuple3s(pd_df[keys].to_numpy())


def move_arg_by_crystalID(crystalID):
    move_arg = FuncArray(p2g[["move_x", "move_y", "move_z"]].to_numpy())
    move_arg.replace_col_with_constant(2, sipm_array.move_z)

    rotate_arg = FuncArray(
        p2g[["rotate_angle_x", "rotate_angle_y", "rotate_angle_z"]].to_numpy()
    )

    return [move_arg.to_list()[crystalID], rotate_arg.to_list()[crystalID]]


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


    def to_plotly(self, idx):
        return [
            *self.counts_plotly_global[idx]
            .map(lambda x: x[0].to_plotly_as_mesh3d(opacity=0.8, color="pink", name="optical photon counts"))
            .array,
            *self.counts_plotly_global[idx]
            .map(lambda x: x[1].to_plotly_as_mesh3d(opacity=0.8, color="pink", name="optical photon counts"))
            .array,
            self.sourcePos[idx].to_plotly(marker=dict(color="red", size=4), name="source position"),
            *self.real_lor[idx].to_plotly_segment(marker=dict(color="gold", size=3), name="real LOR"),
            *self.anger_infered_lor[idx].to_plotly_segment(
                marker=dict(color="blue", size=3), name="anger logic infered LOR"
            ),
            *image_system.to_plotly(),
        ]
