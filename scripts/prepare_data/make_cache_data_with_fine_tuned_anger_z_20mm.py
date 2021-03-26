import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ[
    "DB_CONNECTION"
] = "postgresql://postgres@192.168.1.96:5432/monolithic_crystal"

os.environ[
    "IMAGE_SYSTEM"
] = "/home/zengqinglong/optical_simu/system_50x50x20x80_big_sphere_plus/task_4000_subs/sub.123/Geometry.mac"
os.environ["CRYSTAL_Z"] = "20"
os.environ["CACHED_DATA_DIR"] = "/home/zengqinglong/optical_simu/cached_data"

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

complete_fname = lambda fname: f"""{fname}_CRYSTAL_Z_{os.getenv('CRYSTAL_Z')}_COUNTS_SCALE_VALUE_{os.getenv('COUNTS_SCALE_VALUE')}"""

def messager(mtype, mcontent):
    print(f"""[{mtype.upper()}] {mcontent}""")

def save_as_npy(fname, array):
    with open(fname, "wb") as f:
        np.save(f, array)

def get_full_cached_data_path(fname):
    return '/'.join([os.getenv('CACHED_DATA_DIR'), fname])

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

def print_messate_n_save_to_npy(var_name, var):
    messager('message', f'Making cache data for var: {var_name}')
    full_name = get_full_cached_data_path(f"""{var_name}_experiment_{complete_fname('_'.join(args.experiment_id))}.npy""")
    messager('message', f'Going to save {var_name} to {full_name}.')
    save_as_npy(full_name, var)


messager('message', f'Making cache data for experiment: {args.experiment_id}')
raw_sample = Database().read_sql(get_stmt(args.experiment_id))
c = Counts(raw_sample)



####################################################################################################

def local_z_infer_factory(counts, local_z, statical_func=np.std, polyfit_deg=15):
    counts_statical_feature = counts.map(statical_func)
    sorted_loaclz_vs_statical_feature = np.sort(
        np.stack([
            local_z, 
            counts_statical_feature.array], axis=1), axis=0)
    
    sorted_localz = sorted_loaclz_vs_statical_feature[:,0]
    sorted_statical_feature = sorted_loaclz_vs_statical_feature[:,1]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(
        sorted_statical_feature,
        sorted_localz
    )

    fit_line_x = np.linspace(
        sorted_statical_feature.min(), 
        sorted_statical_feature.max(), 
        len(sorted_loaclz_vs_statical_feature)
    )
    ploy_para = np.polyfit(
        sorted_statical_feature, sorted_localz, deg=polyfit_deg)
    statical_feature_to_local_z = np.poly1d(ploy_para)

    ax.plot(
        fit_line_x,
        statical_feature_to_local_z(fit_line_x)
    )
    return statical_feature_to_local_z, counts_statical_feature


statical_feature_to_local_z, counts_statical_feature = local_z_infer_factory(
    counts = c.sipm_counts_n_position[:,:,:,0],
    local_z = c.gamma_1_local.z,
)


statistical_feature_func = np.std

gamma_1_local_anger = c.anger_infered_lor.fst.batch_to_local(c.crystalIDs[:,0])
gamma_2_local_anger = c.anger_infered_lor.snd.batch_to_local(c.crystalIDs[:,1])

gamma_1_statistical_feature = c.sipm_counts_n_position[:,:,:,0].map(statistical_feature_func)
gamma_2_statistical_feature = c.sipm_counts_n_position[:,:,:,4].map(statistical_feature_func)

gamma_1_statistical_z = statical_feature_to_local_z(gamma_1_statistical_feature.array)
gamma_1_local_anger.z = gamma_1_statistical_z
gamma_2_statistical_z = statical_feature_to_local_z(gamma_2_statistical_feature.array)
gamma_2_local_anger.z = gamma_2_statistical_z

gamma_1_anger = gamma_1_local_anger.batch_to_global(c.crystalIDs[:,0])
gamma_2_anger = gamma_2_local_anger.batch_to_global(c.crystalIDs[:,1])

c.anger_infered_lor = Segment(gamma_1_anger, gamma_2_anger)



####################################################################################################



sipm_counts_n_position = c.sipm_counts_n_position.to_numpy()
print_messate_n_save_to_npy('sipm_counts_n_position_with_statistical_z', sipm_counts_n_position)

anger_infered_lor = c.anger_infered_lor.to_listmode()
print_messate_n_save_to_npy('anger_infered_lor_with_statistical_z', anger_infered_lor)

real_lor = c.real_lor.to_listmode()
print_messate_n_save_to_npy('real_lor_with_statistical_z', real_lor)

source = c.sourcePos.to_numpy().T
print_messate_n_save_to_npy('source_with_statistical_z', source)

