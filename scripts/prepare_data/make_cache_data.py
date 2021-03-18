import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ[
    "DB_CONNECTION"
] = "postgresql://postgres@192.168.1.96:5432/monolithic_crystal"

os.environ[
    "IMAGE_SYSTEM"
] = "/home/zengqinglong/optical_simu/system_50x50x15x80_complicated_phantom/test_simu/Geometry.mac"
os.environ["CRYSTAL_Z"] = "15"
# os.environ["NETWROK_PATH"] = "/home/zengqinglong/.train/2021-02-19_18:46:47"

os.environ["P2G_SQL"] = "select * from pos_local_to_global_view;"
os.environ["COUNTS_SCALE_VALUE"] = '1'

import pandas as pd
import numpy as np
import sympy as sp

from hotpot.geometry.system import (
    SipmArray,
    Hit,
    FuncDataFrame,
    HitsEventIDMapping
)
from hotpot.geometry.primiary import (
    Cartesian3,
    Segment,
    Database, 
    Surface, 
    Trapezoid,
    Box
)
from hotpot.functools import FuncArray
from hotpot.simulation.image_system import ImageSystem, AlbiraImageSystem
from hotpot.simulation.mac import MAC
from hotpot.simulation.sample import SampleWithAnger
from hotpot.counts import Counts, sipm_local_to_global
from hotpot.train import Train
from hotpot.loss import point_line_distance, point_line_distance_with_limitation

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import tensorflow as tf
from uuid import UUID


parser = argparse.ArgumentParser(
    description="Prepare training data for monolithic crystal system LOR training."
)
parser.add_argument("experiment_id", metavar='N', nargs='+', type=str, help="Experiment ID.")
args = parser.parse_args()
experiment_ids = '_'.join(args.experiment_id)

compete_fname = lambda fname: f"""CRYSTAL_Z_{os.getenv('CRYSTAL_Z')}_COUNTS_SCALE_VALUE_{os.getenv('COUNTS_SCALE_VALUE')}_{fname}"""

def save_as_npy(fname, array):
    with open(fname, "wb") as f:
        np.save(f, array)

def get_stmt(experiment_id):
    base = """
SELECT
        cs.*,
        lmwl.gamma_1_x,
        lmwl.gamma_1_y,
        lmwl.gamma_1_z,
        lmwl.gamma_1_local_x,
        lmwl.gamma_1_local_y,
        lmwl.gamma_1_local_z,
        lmwl.gamma_2_x,
        lmwl.gamma_2_y,
        lmwl.gamma_2_z,
        lmwl.gamma_2_local_x,
        lmwl.gamma_2_local_y,
        lmwl.gamma_2_local_z
FROM
        coincidence_sample cs
        JOIN list_mode_with_local lmwl ON (cs. "eventID" = lmwl. "eventID")
        JOIN experiment_coincidence_event ece ON (cs. "eventID" = ece. "eventID")
WHERE
"""
    get_select_stmt = lambda eid: f"""ece.experiment_id = {eid}"""
    if len(experiment_id)==1:
        select_stmt = get_select_stmt(experiment_id[0])
    else:
        select_stmt = " OR ".join([get_select_stmt(i) for i in experiment_id])
    return base+select_stmt


print(args.experiment_id)

raw_sample = Database().read_sql(get_stmt(args.experiment_id))

sipm_counts_n_position = c.sipm_counts_n_position.to_numpy()
anger_infered_lor = c.anger_infered_lor.to_listmode()
real_lor = c.real_lor.to_listmode()
source = c.sourcePos.to_numpy().T


