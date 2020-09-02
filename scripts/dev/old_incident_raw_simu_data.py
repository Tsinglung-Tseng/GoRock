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

ddd = pd.read_sql("""SELECT image_system_id,
       crystal_id,
       c3.x            AS center_x,
       c3.y            AS center_y,
       c3.z            AS center_z,
       rv.x * rv.angle AS rotate_x,
       rv.y * rv.angle AS rotate_y,
       rv.z * rv.angle AS rotate_z,
       detector_center_x,
       detector_center_y,
       detector_center_z
FROM (
         SELECT image_system_id,
                crystal_id,
                array_agg(detector_id order by idx)       as detector_id,
                array_agg(detector_center_x order by idx) as detector_center_x,
                array_agg(detector_center_y order by idx) as detector_center_y,
                array_agg(detector_center_z order by idx) as detector_center_z
         FROM (SELECT image_system_id,
                      crystal_id,
                      array_agg(detector_id order by idy) as detector_id,
                      idx,
                      array_agg(dc.x order by idy)        as detector_center_x,
                      array_agg(dc.y order by idy)        as detector_center_y,
                      array_agg(dc.z order by idy)        as detector_center_z
               FROM image_system.image_system AS i
                        JOIN image_system.crystal AS c USING (image_system_id)
                        JOIN image_system.cartesian3 AS center ON center.cartesian3_id = c.center_cartesian3_id
                        JOIN image_system.detector AS d USING (crystal_id)
                        JOIN image_system.detector_plane2d USING (detector_id)
                        JOIN image_system.cartesian3 AS dc ON d.local_position_cartesian3_id = dc.cartesian3_id

               GROUP BY image_system_id, crystal_id, idx) AS t
         GROUP BY image_system_id, crystal_id
         ORDER BY image_system_id, crystal_id
     ) AS t
         JOIN image_system.crystal c using (image_system_id, crystal_id)
         JOIN (SELECT r.rotate_vector3_id,
                      sin(u.theta) * cos(u.phi) AS x,
                      sin(u.theta) * sin(u.phi) AS y,
                      cos(u.theta)              AS z,
                      r.angle
               FROM image_system.rotate_vector3 r
                        JOIN image_system.unit_vector3 u USING (unit_vector3_id)) AS rv
              ON c.rotate_vector3_id = rv.rotate_vector3_id
         JOIN image_system.cartesian3 c3 ON c.center_cartesian3_id = c3.cartesian3_id
WHERE image_system_id = 5;

""", con=Database().engine())

np.array(ddd['detector_center_x'].tolist()).flatten().shape

# sipm local coordinate
go.Figure(Cartisian3(
    np.array(ddd['detector_center_x'].tolist()).flatten(),
    np.array(ddd['detector_center_y'].tolist()).flatten(),
    np.array(ddd['detector_center_z'].tolist()).flatten()
).to_plotly())

# crystal central coordinate
go.Figure(Cartisian3(
    np.array(ddd['center_x'].tolist()).flatten(),
    np.array(ddd['center_y'].tolist()).flatten(),
    np.array(ddd['center_z'].tolist()).flatten()
).to_plotly())