from .primiary import Cartesian3
import numpy as np
import pandas as pd
import tensorflow as tf
from ..database import Database


def most_photon_crystal(a_single):
    return a_single.groupby("crystalID").count().idxmax()[0]


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

    def conincidence_sample(self, sipm: SipmArray = SipmArray()):
        def get_conincidence_hist(hits, eventID, bins=sipm.bins):
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
                sample["counts"] = get_conincidence_hist(hits, eventID, bins=bins)
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
                    .numpy(),
                    sipm.local_pos.move(gamma_2_move_args[:3])
                    .rotate_using_rotate_matrix(rotation_matrix_y(gamma_2_move_args[3]))
                    .fmap(lambda i: tf.reshape(i, (sipm.bins, sipm.bins)))
                    .to_tensor()
                    .numpy(),
                ]
                return sample

            return list(
                _assemble_single_sample(eventID) for eventID in self.coincidence.ids
            )

        return assemble_samples(self.df)

    def single_sample(self, sipm: SipmArray = SipmArray()):
        def get_single_hist(hits, eventID, bins=sipm.bins):
            gamma = hits[hits.eventID == eventID]

        return Hit(gamma_2).hist2d()
