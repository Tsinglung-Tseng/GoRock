import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DB_CONNECTION"] ="postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
os.environ["IMAGE_SYSTEM"] = "/home/zengqinglong/optical_simu/system_50x50x20x80_valid_6p_correct/task_400_subs/sub.123/Geometry.mac"
os.environ["CRYSTAL_Z"] = "20"
os.environ["NETWROK_PATH"] = "/home/zengqinglong/.train/2021-02-19_18:46:47"
os.environ["P2G_SQL"] = "select * from pos_local_to_global_view;"
os.environ["COUNTS_SCALE_VALUE"] = '1'

import pandas as pd
import numpy as np
import sympy as sp

# from hotpot.geometry.system import (
    # SipmArray,
    # Hit,
    # FuncDataFrame,
    # HitsEventIDMapping
# )
from hotpot.geometry.primiary import (
    # Cartesian3,
    # Segment,
    Database, 
    # Surface, 
    # Trapezoid,
    # Box
)
# from hotpot.functools import FuncArray
# from hotpot.simulation.image_system import ImageSystem, AlbiraImageSystem
# from hotpot.simulation.mac import MAC
# from hotpot.simulation.sample import SampleWithAnger
# from hotpot.counts import Counts, sipm_local_to_global
# from hotpot.train import Train
# from hotpot.loss import point_line_distance, point_line_distance_with_limitation

# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from uuid import UUID
# from hotpot.counts import image_system, p2g, sipm_array, move_arg_by_crystalID
# to_plotly_surface = lambda z: go.Figure(go.Surface(z=z))

raw_sample = Database().read_sql("""
        SELECT table_name
  FROM information_schema.tables;
""")

print(raw_sample)
