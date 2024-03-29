from hotpot.geometry.primiary import Cartesian3
from hotpot.database import Database
from hotpot.functools import FuncDataFrame
from hotpot.simulation.image_system import ImageSystem
from hotpot.geometry.system import SipmArray
from scipy.ndimage.filters import uniform_filter1d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import os
from hotpot.functools import FuncList
import re


class ModelTuner:
    def __init__(self, sample):
        self.sample = sample
        self.p2g = Database().read_sql("""select * from pos_local_to_global_view;""")
        #         self.gate_rm = FuncDataFrame(pd.read_sql("""select * from rotate_rc;""", con=Database().engine()))
        self.ims = ImageSystem.from_file(
            "/home/zengqinglong/optical_simu/5/jiqun_test_ja_9_2mm/macro/Geometry.mac"
        )

    def p2g_by_crystalID(self, crystalID):
        return [
            self.p2g[["move_x", "move_y", "move_z"]].iloc[crystalID].to_list(),
            self.p2g[["rotate_angle_x", "rotate_angle_y", "rotate_angle_z"]]
            .iloc[crystalID]
            .to_list(),
        ]

    @staticmethod
    def df_to_car3(df):
        tmp = np.array(df.to_records(index=False).tolist()).T
        return Cartesian3(x=tmp[0], y=tmp[1], z=tmp[2])

    def get_lor_Seg_by_eventID(self, eventID):
        class Segment:
            def __init__(self, pair):
                self.pair = pair

            def to_plotly(self):
                x = self.pair.reshape([2, 3]).T[0]
                y = self.pair.reshape([2, 3]).T[1]
                z = self.pair.reshape([2, 3]).T[2]
                return go.Scatter3d(x=x, y=y, z=z)

        return Segment(
            FuncDataFrame(self.sample)
            .where(eventID=eventID)
            .df[
                [
                    "gamma_1_x",
                    "gamma_1_y",
                    "gamma_1_z",
                    "gamma_2_x",
                    "gamma_2_y",
                    "gamma_2_z",
                ]
            ]
            .iloc[0]
            .to_numpy()
        )

    def counts_2_mesh3d(self, counts, crystalID):
        tmp = SipmArray().local_pos
        tmp.z = (tmp.z + counts.flatten()) / 10

        tmp = tmp.move(self.p2g_by_crystalID(crystalID)[0]).rotate_ypr(
            self.p2g_by_crystalID(crystalID)[1]
        )
        #     tmp = (
        #         tmp
        #         .move(gate_rm.where(crystalID=crystalID).df.C.tolist()[0])
        #         .left_matmul(gate_rm.where(crystalID=crystalID).df.R.tolist()[0])
        #     )

        return go.Mesh3d(x=tmp.x, y=tmp.y, z=tmp.z, color="pink", opacity=0.8)

    def to_plotly(self, eID):
        gamma_1_counts, gamma_2_counts = np.array(
            FuncDataFrame(self.sample).where(eventID=eID).df.counts.to_list()
        )[0]
        gamma_1_sipm, gamma_2_sipm = np.array(
            FuncDataFrame(self.sample).where(eventID=eID).df.sipm_center_pos.to_list()
        )[0]
        gamma_1_crystalID, gamma_2_crystalID = (
            FuncDataFrame(self.sample).where(eventID=eID).df.crystalID.to_list()[0]
        )
        return [
            *self.ims.to_plotly(),
            ModelTuner.df_to_car3(
                self.sample[["sourcePosX", "sourcePosY", "sourcePosZ"]]
            ).to_plotly(marker=dict(size=1)),
            self.get_lor_Seg_by_eventID(eID).to_plotly(),
            self.counts_2_mesh3d(gamma_1_counts, gamma_1_crystalID),
            self.counts_2_mesh3d(gamma_2_counts, gamma_2_crystalID),
        ]


def pattern_filter_factory(pattern):
    def get_file_index(fname):
        return int(fname.split(".")[0].split("_")[-1])

    pattern_filter = lambda files, work_dir: (
        FuncList(
            sorted(
                FuncList(files)
                .map(lambda i: [re.findall(pattern, i), i])
                .filter(lambda i: i[0] != [])
                .map(lambda i: [get_file_index(i[1]), i[1]])
                .to_list(),
                key=lambda i: i[0],
            )
        )
        .map(lambda i: "/".join([work_dir, i[-1]]))
        .to_list()
    )
    return pattern_filter


def list_of_array_file_to_array(files):
    npload = lambda fname: np.load(fname)
    return np.concatenate([npload(i) for i in files])


class Train:
    def __init__(self, train_id):
        self.train_id = train_id

    @property
    def work_dir(self):
        return (
            Database()
            .read_sql(
                f"""
        SELECT
            work_dir
        FROM
            train
        WHERE
            id = {self.train_id};"""
            )
            .to_numpy()[0][0]
        )

    @property
    def all_files_on_work_dir(self):
        return list(os.walk(self.work_dir))[0][2]

    @property
    def vars_train(self):
        return pattern_filter_factory("^vars_train_on_.*\.h5$")(
            self.all_files_on_work_dir, self.work_dir
        )

    @property
    def loss_files(self):
        return pattern_filter_factory("^loss_.*\.npy$")(
            self.all_files_on_work_dir, self.work_dir
        )

    @property
    def val_loss_files(self):
        return pattern_filter_factory("^val_loss_.*\.npy$")(
            self.all_files_on_work_dir, self.work_dir
        )

    @property
    def net_infered(self):
        return pattern_filter_factory("^net_infered.*\.npy$")(
            self.all_files_on_work_dir, self.work_dir
        )

    @property
    def loss(self):
        return list_of_array_file_to_array(self.loss_files)

    @property
    def val_loss(self):
        return list_of_array_file_to_array(self.val_loss_files)

    def view_loss(self, ma_window=50, show_from=100):
        epoch_ratio = int(len(self.loss) / len(self.val_loss))

        val_loss = uniform_filter1d(self.val_loss[show_from:], ma_window)
        loss = uniform_filter1d(self.loss[::epoch_ratio][show_from:], ma_window)
        x_seq = np.arange(len(loss))

        # fig = go.Figure()

        # fig.add_trace(
            # go.Scatter(
                # x=x_seq, y=val_loss, name="Validation Loss", line=dict(color="blue")
            # )
        # )

        # fig.add_trace(
            # go.Scatter(x=x_seq, y=loss, name="Train Loss", line=dict(color="red"))
        # )

        # fig.show()
        return [
            go.Scatter(x=x_seq, y=val_loss, name=f"Train {self.train_id} Validation Loss", line=dict(color="blue")),
            go.Scatter(x=x_seq, y=loss, name=f"Train {self.train_id} Train Loss", line=dict(color="red"))
        ]

    def model_on_step(self):
        return pattern_filter_factory("^vars_train_on_.*\.h5$")(
            self.all_files_on_work_dir, self.work_dir
        )
