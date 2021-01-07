import argparse
from hotpot.simulation.sample import SampleWithAnger
from hotpot.geometry.primiary import Cartesian3
from hotpot.database import Database
from hotpot.functools import FuncArray
import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["DB_CONNECTION"] ="postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
os.environ["PICLUSTER_DB"] ="postgresql://picluster@192.168.1.96:5432/picluster"


parser = argparse.ArgumentParser(
    description="Prepare training data for monolithic crystal system LOR training."
)
parser.add_argument("experiment_id", type=str, help="Experiment ID.")
# parser.add_argument("dataset_type", type=str, help="Train or validation.")


def save_as_npy(fname, array):
    with open(fname, 'wb') as f:
        np.save(f, array)


args = parser.parse_args()

stmt = f"""
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
	ece.experiment_id = {args.experiment_id};
"""

sample_df = Database().read_sql(stmt)

swa = SampleWithAnger(sample_df)
sipm_counts_n_position = tf.cast(
    FuncArray(swa.gamma_1_counts)
    .concatenate_with(FuncArray(swa.sipm_center_pos.shrink((1,2)).rollaxis(1,4).array[:,:,:,:3]), axis=3)
    .concatenate_with(FuncArray(tf.transpose(swa.gamma_2_counts, perm=[0,2,1,3])/10), axis=3)
    .concatenate_with(FuncArray(swa.sipm_center_pos.shrink((1,2)).rollaxis(1,4).array[:,:,:,3:]), axis=3)
    .to_tensor()
    , dtype=tf.float32
)

anger_infered = tf.cast(
    FuncArray(swa.gamma_1_anger_global)
    .concatenate_with(FuncArray(swa.gamma_2_anger_global), axis=1)
    .to_tensor()
    , dtype=tf.float32
)

source_position = tf.cast(FuncArray(swa.sample_df[['sourcePosX','sourcePosY','sourcePosZ']]).to_tensor(), tf.float32)

save_as_npy(f"sipm_counts_n_position_experiment_{args.experiment_id}.npy", sipm_counts_n_position)
save_as_npy(f"anger_infered_experiment_{args.experiment_id}.npy", anger_infered)
save_as_npy(f"source_position_experiment_{args.experiment_id}.npy", source_position)
