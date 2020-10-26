from .primiary import Cartesian3
import numpy as np
import pandas as pd
import tensorflow as tf
from ..database import Database


def most_photon_crystal(a_single):
    return int(a_single.groupby("crystalID").count().idxmax()[0])


def move_arg_of_crystal(crystalID):
    return (
        Database()
        .read_sql(
            f"""SELECT move_x, move_y, move_z, rotate_angle_y
FROM pos_local_to_global
WHERE crystal_id={crystalID};"""
        )
        .iloc[0]
    )


def rotation_matrix_y(angle):
    return tf.convert_to_tensor(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


class FuncDataFrame:
    def __init__(self, df):
        self.df = df

    def where(self, **kwargs):
        if len(kwargs) != 1:
            raise ValueError("where clause support one condition at once!")
        for key, value in kwargs.items():
            return FuncDataFrame(self.df[self.df[key] == value])

    def filter(self, key_list):
        return FuncDataFrame(self.df[key_list])


class Crystal:
    # TODO: incase different size of crystal is needed
    pass


class SipmArray:
    def __init__(self, bins=16):
        self.bins = bins
        self.crystal_x = 50
        self.crystal_y = 50
        self.crystal_z = 15

        self.sipm_vertex_x = np.linspace(
            self.crystal_x / 2 - self.crystal_x,
            self.crystal_x / 2,
            self.bins + 1,
        )
        self.sipm_vertex_y = np.linspace(
            self.crystal_y / 2 - self.crystal_y,
            self.crystal_y / 2,
            self.bins + 1,
        )

        self.sipm_center_x = (
            self.sipm_vertex_x[:-1]
            - (self.sipm_vertex_x[:-1] - self.sipm_vertex_x[1:]) / 2
        )
        self.sipm_center_y = (
            self.sipm_vertex_y[:-1]
            - (self.sipm_vertex_y[:-1] - self.sipm_vertex_y[1:]) / 2
        )

        self.sipm_z = self.crystal_z / 2

        self.sipm_vertexs_x, self.sipm_vertexs_y = np.meshgrid(
            self.sipm_vertex_x, self.sipm_vertex_y
        )
        self.sipm_vertexs_z = np.full((self.bins + 1, self.bins + 1), self.sipm_z)

        self.sipm_centers_x, self.sipm_centers_y = np.meshgrid(
            self.sipm_center_x, self.sipm_center_y
        )
        self.sipm_centers_z = np.full((self.bins, self.bins), self.sipm_z)

    @property
    def local_pos(self):
        return Cartesian3(
            self.sipm_centers_x.flatten(),
            self.sipm_centers_y.flatten(),
            self.sipm_centers_z.flatten(),
        )

    def to_plotly(self):
        return (
            Cartesian3(
                self.sipm_centers_x.flatten(),
                self.sipm_centers_y.flatten(),
                self.sipm_centers_z.flatten(),
            ).to_plotly(),
        )


class Hit:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def ids(self):
        return self.df["eventID"].unique()

    @property
    def single(self):
        return Hit(
            self.df.groupby("eventID").filter(
                lambda event: len(event["parentID"].unique()) == 2
            )
        )

    def single_ids(self):
        return self.single.ids

    @property
    def coincidence(self):
        return Hit(
            self.df.groupby("eventID").filter(
                lambda event: len(event["parentID"].unique()) == 3
            )
        )

    def hist2d(self, sipm: SipmArray = SipmArray()):
        return np.histogram2d(self.df.localPosX, self.df.localPosY, bins=sipm.bins)[
            0
        ].tolist()

    def coincidence_sample(self, sipm: SipmArray = SipmArray()):
        def get_coincidence_hist(hits, eventID, bins=sipm.bins):
            gamma_1 = hits[hits.eventID == eventID][
                hits[hits.eventID == eventID].photonID == 1
            ]
            gamma_2 = hits[hits.eventID == eventID][
                hits[hits.eventID == eventID].photonID == 2
            ]

            return [
                Hit(gamma_1).hist2d(),
                Hit(gamma_2).hist2d(),
            ]

        def assemble_samples(hits, bins=sipm.bins):
            def _assemble_single_sample(eventID):
                sample = {}
                sample["eventID"] = eventID
                sample = {
                    **sample,
                    **dict(
                        hits[hits["eventID"] == eventID][
                            ["sourcePosX", "sourcePosY", "sourcePosZ"]
                        ].iloc[0]
                    ),
                }
                sample["counts"] = get_coincidence_hist(hits, eventID, bins=bins)
                sample["crystalID"] = [
                    most_photon_crystal(
                        hits[hits["eventID"] == eventID][
                            hits[hits["eventID"] == eventID]["photonID"] == 1
                        ]
                    ),
                    most_photon_crystal(
                        hits[hits["eventID"] == eventID][
                            hits[hits["eventID"] == eventID]["photonID"] == 2
                        ]
                    ),
                ]

                gamma_1_move_args = move_arg_of_crystal(sample["crystalID"][0])
                gamma_2_move_args = move_arg_of_crystal(sample["crystalID"][1])

                sample["sipm_center_pos"] = [
                    sipm.local_pos.move(gamma_1_move_args[:3])
                    .rotate_using_rotate_matrix(rotation_matrix_y(gamma_1_move_args[3]))
                    .fmap(lambda i: tf.reshape(i, (sipm.bins, sipm.bins)))
                    .to_tensor()
                    .numpy()
                    .tolist(),
                    sipm.local_pos.move(gamma_2_move_args[:3])
                    .rotate_using_rotate_matrix(rotation_matrix_y(gamma_2_move_args[3]))
                    .fmap(lambda i: tf.reshape(i, (sipm.bins, sipm.bins)))
                    .to_tensor()
                    .numpy()
                    .tolist(),
                ]

                # sample["PhotoElectricPosX"]
                return sample

            return list(
                _assemble_single_sample(eventID) for eventID in self.coincidence.ids
            )

        return pd.DataFrame(assemble_samples(self.df))

    def single_sample(self, sipm: SipmArray = SipmArray()):
        def get_single_hist(hits, eventID, bins=sipm.bins):
            gamma = hits[hits.eventID == eventID]

        return Hit(gamma_2).hist2d()

    def commit_coincidentce_sample_to_database(self, experiment_id):
        # coincidence_sample = [tuple(s) for s in self.coincidence.coincidence_sample().to_numpy()]
        experiment_gamma = [
            tuple((experiment_id, *row))
            for row in FuncDataFrame(self.coincidence.df)
            .where(processName="PhotoElectric")
            .filter(["eventID", "photonID"])
            .df.to_numpy()
        ]
        with Database().cursor() as (conn, cur):
            # cur.executemany(
            # "INSERT INTO coincidence_sample (eventID,sourcePosX,sourcePosY,sourcePosZ,counts,crystalID,sipm_center_pos) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            # coincidence_sample,
            # )
            cur.executemany(
                """INSERT INTO experiment_gamma ("experiment_id","eventID","photonID") VALUES (%s,%s,%s)""",
                experiment_gamma,
            )
            conn.commit()
