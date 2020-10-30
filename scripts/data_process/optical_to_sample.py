import pandas as pd
import numpy as np
from hotpot.geometry.system import Crystal, SipmArray, Hit, FuncDataFrame, HitsEventIDMapping
import matplotlib.pyplot as plt
from hotpot.geometry.primiary import Cartesian3, Database
import plotly.graph_objects as go
import os
import argparse


os.environ["DB_CONNECTION"] = "postgresql://zengqinglong@database/monolithic_crystal"

# parse args
parser = argparse.ArgumentParser(
    description="Count optical photon, aggragate optical simulation data to sample."
)
parser.add_argument("path", type=str, help="Path of imput hits csv file.")
parser.add_argument(
    "experiment_id", type=str, help="Experiment id to which this hits belongs."
)

args = parser.parse_args()


# def commit_to_experiment(experiment_id, coincidence_sample):
#     records = [
#         tuple((experiment_id, eventID)) for eventID in coincidence_sample["eventID"]
#     ]
#     with Database().cursor() as (conn, cur):
#         cur.executemany(
#             """INSERT INTO experiment_coincidence_event ("experiment_id", "eventID") VALUES (%s,%s)""",
#             records,
#         )
#         conn.commit()


hits = pd.read_csv(args.path)
h = Hit(hits)

# replace number event id with uuid id
HitsEventIDMapping.build(h.df).do_replace(h.df)

# h.set_experiment_id(args.experiment_id)

# coincidence_sample table and experiment_coincidence_event table
h.gamma_hits.commit('gamma_hits')
h.commit_coincidentce_sample_to_database(args.experiment_id)
