import tensorflow as tf
import pandas as pd

from ..database import Database

from ..geometry.primiary import (
    Cartisian3,
    get_source,
    Pair,
    PairCartisian3,
    split_raw_df_into_even_odd_pairs,
)


class Dataset:
    """
    这里需要的索引很灵活
    
    首先需要区分训练测试集
    dataset :: Dataset
    dataset.train :: SubDataset | Pair

    索引数据时，
    既需要：
    dataset.train.gamma_incident_local :: Cartisian3 | Pair
    dataset.train.gamma_incident_local.fst :: Cartisian3

    也需要：
    dataset.train.fst :: SubDataset
    dataset.train.fst.gamma_incident_local :: Cartisian3

    >>> pc = Dataset(12)
    >>> go.Figure([pc.train.gamma_incident_local.fst.to_plotly(), pc.train.gamma_incident_local.snd.to_plotly()])
    or
    >>> go.Figure(pc.valid.gamma_incident_local.fst.to_plotly())
    """

    def __init__(self, dataset_id):
        self.dataset_id = dataset_id
        self.raw = get_dataset_by_id(dataset_id)
        self.train = SubDataset(self.raw, "train")
        self.test = SubDataset(self.raw, "test")
        self.valid = SubDataset(self.raw, "valid")

    def __repr__(self):
        return f"""Dataset: <dataset_id = {self.dataset_id}>"""


class SubDataset:
    def __init__(self, raw: pd.DataFrame, division):
        self.division = division
        self.raw = raw[self.division]
        self.source = Cartisian3.from_pattern(self.raw, "source_")
        self.gamma_incident_local = PairCartisian3(self.raw, "g")
        self.fst, self.snd = split_raw_df_into_even_odd_pairs(self.raw)

    def __repr__(self):
        return f"SubDataset: <division = {self.division}>"


class InMemoryDataset:
    pass


def query_to_pd(stmt):
    with Database().cursor() as (conn, cur):
        return pd.read_sql_query(stmt, con=conn)


def get_dataset_by_id(dataset_id):
    stmt = (
        lambda dataset_id, dataset_type: f"""SELECT
            emitter_id,
            photon_id,
            gamma_incident_id,
            source_x, 
            source_y,
            source_z,
            local_pos_x AS gx,
            local_pos_y AS gy,
            local_pos_z AS gz,
            counts,
            crystal_id
        FROM (
            SELECT
                emitter_id
            FROM
                gate.dataset_split AS ds
                JOIN gate.emitter_meta AS e USING (emitter_id)
            WHERE
                dataset_id = {dataset_id}
                AND dataset_type = '{dataset_type}'
            ORDER BY
                random()
        ) AS t
            JOIN gate.emitter_position AS ep USING (emitter_id)
            JOIN gate.gamma_incident_meta AS g USING (emitter_id)
            JOIN gate.gamma_incident_position AS gp USING (gamma_incident_id)
            JOIN gate.gamma_incident_count AS gc USING (gamma_incident_id)
        ORDER BY
            emitter_id, photon_id;"""
    )
    return {
        "train": query_to_pd(stmt(dataset_id, "train")),
        "test": query_to_pd(stmt(dataset_id, "test")),
        "valid": query_to_pd(stmt(dataset_id, "valid")),
    }
