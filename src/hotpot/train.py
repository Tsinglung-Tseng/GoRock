from hotpot.geometry.primiary import Cartesian3
from hotpot.database import Database
from hotpot.functools import FuncDataFrame
from hotpot.simulation.image_system import ImageSystem
from hotpot.geometry.system import SipmArray
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import plotly.graph_objects as go


class ModelTuner:

    def __init__(self, sample):
        self.sample = sample
        self.p2g = Database().read_sql("""select * from pos_local_to_global_view;""")
#         self.gate_rm = FuncDataFrame(pd.read_sql("""select * from rotate_rc;""", con=Database().engine()))
        self.ims = ImageSystem.from_file('/home/zengqinglong/optical_simu/5/jiqun_test_ja_9_2mm/macro/Geometry.mac')

    def p2g_by_crystalID(self, crystalID):
        return [
            self.p2g[['move_x','move_y','move_z']].iloc[crystalID].to_list(),
            self.p2g[['rotate_angle_x','rotate_angle_y','rotate_angle_z']].iloc[crystalID].to_list()
        ]

    @staticmethod
    def df_to_car3(df):
        tmp = np.array(df.to_records(index=False).tolist()).T
        return Cartesian3(
            x=tmp[0],
            y=tmp[1],
            z=tmp[2]
        )

    def get_lor_Seg_by_eventID(self, eventID):
        class Segment:
            def __init__(self, pair):
                self.pair = pair

            def to_plotly(self):
                x=self.pair.reshape([2,3]).T[0]
                y=self.pair.reshape([2,3]).T[1]
                z=self.pair.reshape([2,3]).T[2]
                return go.Scatter3d(x=x,y=y,z=z)

        return Segment(FuncDataFrame(self.sample).where(eventID=eventID).df[['gamma_1_x','gamma_1_y','gamma_1_z','gamma_2_x','gamma_2_y','gamma_2_z']].iloc[0].to_numpy())

    def counts_2_mesh3d(self, counts, crystalID):
        tmp = SipmArray().local_pos
        tmp.z = (tmp.z+counts.flatten())/10

        tmp = tmp.move(self.p2g_by_crystalID(crystalID)[0]).rotate_ypr(self.p2g_by_crystalID(crystalID)[1])
    #     tmp = (
    #         tmp
    #         .move(gate_rm.where(crystalID=crystalID).df.C.tolist()[0])
    #         .left_matmul(gate_rm.where(crystalID=crystalID).df.R.tolist()[0])
    #     )

        return go.Mesh3d(
            x=tmp.x,
            y=tmp.y,
            z=tmp.z,
            color='pink', opacity=0.8
        )

    def to_plotly(self, eID):
        gamma_1_counts, gamma_2_counts = np.array(FuncDataFrame(self.sample).where(eventID=eID).df.counts.to_list())[0]
        gamma_1_sipm, gamma_2_sipm = np.array(FuncDataFrame(self.sample).where(eventID=eID).df.sipm_center_pos.to_list())[0]
        gamma_1_crystalID, gamma_2_crystalID = FuncDataFrame(self.sample).where(eventID=eID).df.crystalID.to_list()[0]
        return [
            *self.ims.to_plotly(),
            ModelTuner.df_to_car3(self.sample[['sourcePosX','sourcePosY','sourcePosZ']]).to_plotly(marker=dict(size=1)),
            self.get_lor_Seg_by_eventID(eID).to_plotly(),
            self.counts_2_mesh3d(gamma_1_counts, gamma_1_crystalID),
            self.counts_2_mesh3d(gamma_2_counts, gamma_2_crystalID)
        ]
