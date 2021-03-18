from hotpot.simulation.sample import SampleWithAnger
import numpy as np
from .functools import FuncArray
from .database import Database


class AngerDataSet:
    def __init__(self, sample_df):
        self.sample_dfmple_df = sample_df
        self.swa = SampleWithAnger(sample_df)

    @property
    def sipm_counts_n_position(self):
        return tf.cast(
            FuncArray(self.swa.gamma_1_counts)
            .concatenate_with(
                FuncArray(
                    self.swa.sipm_center_pos.shrink((1, 2))
                    .rollaxis(1, 4)
                    .array[:, :, :, :3]
                ),
                axis=3,
            )
            .concatenate_with(
                FuncArray(
                    tf.transpose(self.swa.gamma_2_counts, perm=[0, 2, 1, 3]) / 10
                ),
                axis=3,
            )
            .concatenate_with(
                FuncArray(
                    self.swa.sipm_center_pos.shrink((1, 2))
                    .rollaxis(1, 4)
                    .array[:, :, :, 3:]
                ),
                axis=3,
            )
            .to_tensor(),
            dtype=tf.float32,
        )

    @property
    def anger_infered(self):
        return tf.cast(
            FuncArray(self.swa.gamma_1_anger_global)
            .concatenate_with(FuncArray(self.swa.gamma_2_anger_global), axis=1)
            .to_tensor(),
            dtype=tf.float32,
        )

    @property
    def source_position(self):
        return tf.cast(
            FuncArray(
                self.swa.sample_df[["sourcePosX", "sourcePosY", "sourcePosZ"]]
            ).to_tensor(),
            tf.float32,
        )


class CachedData:
    def __init__(self, dataset_id):
        self.dataset_id = dataset_id

    @property
    def sipm_counts_n_position(self):
        return np.load(
            Database()
            .read_sql(
                f"""
        SELECT
            sipm_counts_n_position
        FROM
            experiment_cahced_data
        WHERE
            id = {self.dataset_id};"""
            )
            .to_numpy()[0][0]
        )

    @property
    def anger_infered(self):
        return np.load(
            Database()
            .read_sql(
                f"""
        SELECT
            anger_infered
        FROM
            experiment_cahced_data
        WHERE
            id = {self.dataset_id};"""
            )
            .to_numpy()[0][0]
        )

    @property
    def source_position(self):
        return np.load(
            Database()
            .read_sql(
                f"""
        SELECT
            source_position
        FROM
            experiment_cahced_data
        WHERE
            id = {self.dataset_id};"""
            )
            .to_numpy()[0][0]
        )

    @property
    def real_lor(self):
        return np.load(
            Database()
            .read_sql(
                f"""
        SELECT
            real_lor
        FROM
            experiment_cahced_data
        WHERE
            id = {self.dataset_id};"""
            )
            .to_numpy()[0][0]
        )


class DBDataset:
    def __init__(self, table_name):
        self.table_name = table_name
        self._data = None

    def build(self):
        self._data = Database().read_sql(f"select * from {self.table_name};").to_numpy()

    @property
    def data(self):
        if self._data is not None:
            return self._data
        else:
            self.build()
            return self.data
