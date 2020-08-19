import tensorflow as tf
import pandas as pd

from ..database import Database


class Dataset:
    pass


class InMemoryDataset:
    pass


def query_to_pd(stmt):
    with Database().cursor() as (conn, cur):
        return pd.read_sql_query(stmt, con=conn)


def get_dataset_by_id(dataset_id):
    stmt = lambda dataset_id, dataset_type: f"""SELECT
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
    return {
        'train': query_to_pd(stmt(dataset_id, 'train')),
        'test': query_to_pd(stmt(dataset_id, 'test')),
        'valid': query_to_pd(stmt(dataset_id, 'valid'))
    }


