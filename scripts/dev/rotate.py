import tensorflow as tf
from hotpot.utils.tf_gpu import USEGPU
USEGPU(0)

from tables import open_file
import h5py
import pandas as pd
import psycopg2
import numpy as np

import plotly.graph_objects as go
import ipyvolume as ipv

from hotpot.database import Database
from hotpot.dl_network.dataset import query_to_pd, get_dataset_by_id, Dataset
from hotpot.geometry.primiary import Cartisian3, get_source

from hotpot.math.rotate import Vector3, Quaternion, Rotate3D, UnitVector

import os
os.environ["DB_CONNECTION"] ="postgresql://postgres:postgres@192.168.1.185:54322/incident"

if __name__ == "__main__":
    
    rotate = pd.read_sql(
    """
    SELECT
        c.image_system_id,
        ims.description,
        c.crystal_id,
        -- 	c.rotate_vector3_id,
        uv3.theta,
        uv3.phi,
        rv3.angle
    FROM
        image_system.crystal c
        LEFT JOIN image_system.image_system ims ON c.image_system_id = ims.image_system_id
        JOIN image_system.rotate_vector3 rv3 ON c.rotate_vector3_id = rv3.rotate_vector3_id
        JOIN image_system.unit_vector3 uv3 ON rv3.unit_vector3_id = uv3.unit_vector3_id
        WHERE ims.image_system_id=5;
    """,
    con=Database().engine()
    )

    uv3 = UnitVector(rotate['theta'], rotate['phi'])

    rotate_quaternion = Quaternion.from_axis_angle(uv3, rotate['angle'])

    rm = rotate_quaternion.to_rotation_matrix()

    print(rm.shape)