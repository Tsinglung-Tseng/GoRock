from hotpot.geometry.primiary import Cartesian3
import tensorflow as tf

tf.keras.backend.set_image_data_format('channels_last')

import numpy as np
import pandas as pd

import plotly.graph_objects as go

from hotpot.functools import FuncArray, FuncDataFrame, FuncNNLayer
from hotpot.log import LossHistory
from hotpot.loss import point_line_distance, point_line_distance_with_limitation

import os
from hotpot.database import Database
os.environ["DB_CONNECTION"] ="postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
os.environ["PICLUSTER_DB"] ="postgresql://picluster@192.168.1.96:5432/picluster"

sipm_counts_n_position = tf.cast(np.load('./sipm_counts_n_position_experiment_12.npy'), tf.float32)
anger_infered = tf.cast(np.load('./anger_infered_experiment_12.npy'), tf.float32)
source_position = tf.cast(np.load('./source_position_experiment_12.npy'), tf.float32)

stacked_counts = tf.keras.Input(shape=(16,16,8), name='sipm_counts_n_position')
anger = tf.keras.Input(shape=(6), name='anger_infered')

res_component = (
    FuncNNLayer(stacked_counts)
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Flatten())
    .append_next_layer(tf.keras.layers.Dense(1024))
    .append_next_layer(tf.keras.layers.Dense(1024))
    .append_next_layer(tf.keras.layers.Dense(512))
    .append_next_layer(tf.keras.layers.Dense(512))
    .append_next_layer(tf.keras.layers.Dense(256))
    .append_next_layer(tf.keras.layers.Dense(256))
    .append_next_layer(tf.keras.layers.Dense(128))
    .append_next_layer(tf.keras.layers.Dense(128))
    .append_next_layer(tf.keras.layers.Dense(64))
    .append_next_layer(tf.keras.layers.Dense(64))
    .append_next_layer(tf.keras.layers.Dense(32))
    .append_next_layer(tf.keras.layers.Dense(32))
    .append_next_layer(tf.keras.layers.Dense(16))
    .append_next_layer(tf.keras.layers.Dense(16))
    .append_next_layer(tf.keras.layers.Dense(6))
)


outputs = res_component.layer + anger

model = tf.keras.Model(inputs=[stacked_counts, anger], outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.0001, momentum=0.1, nesterov=False, name='SGD'
    ),
    loss=point_line_distance_with_limitation,
)
history = LossHistory()

sipm_counts_n_position_valid = tf.cast(np.load('/home/zengqinglong/jupyters/monolithicCrystal/sipm_counts_n_position_8.npy'), tf.float32)
anger_infered_valid = tf.cast(np.load('/home/zengqinglong/jupyters/monolithicCrystal/anger_infered_8.npy'), tf.float32)
source_position_valid = tf.cast(np.load('/home/zengqinglong/jupyters/monolithicCrystal/source_position_8.npy'), tf.float32)

for i in range(100):
    model.fit(
        {"input_1": sipm_counts_n_position, "intput_2": anger_infered},
        source_position,
        batch_size=128,
        epochs=100,
        validation_split=0.1,
        callbacks=[history]
    )
    model.save_weights(f"vars_step_{i}.h5")
    net_infered_8 = model([sipm_counts_n_position_valid, anger_infered_valid]) 
    with open(f'net_infered_8_step_{i}.npy', 'wb') as f:
        np.save(f, net_infered_8)
    with open(f'loss_step_{i}.npy', 'wb') as f:
        np.save(f, np.array(history.losses))
    with open(f'val_loss_step_{i}.npy', 'wb') as f:
        np.save(f, np.array(history.val_losses))


# from hotpot.sample import FuncArray
# import argparse
# from hotpot.geometry.primiary import Cartesian3
# from hotpot.database import Database
# import pandas as pd
# import numpy as np
# from hotpot.geometry.system import Crystal, SipmArray, Hit
# import os
# import uuid
# 
# import matplotlib.pyplot as plt
# 
# os.environ[
#     "DB_CONNECTION"
# ] = "postgresql://zengqinglong@192.168.1.96:5432/monolithic_crystal"
# os.environ["PICLUSTER_DB"] = "postgresql://picluster@192.168.1.96:5432/picluster"
# sample_stmt = """
#         SELECT ts.*
#     FROM train_sample ts
#         JOIN experiment_coincidence_event ece ON (ts."eventID"=ece."eventID")
#         JOIN experiments e ON (ece.experiment_id=e.id)
#         WHERE ts.gamma_1_x is not NULL AND experiment_id = 8;
# """
# 
# #     '''SELECT
# #         cs."eventID",
# #         cs."sourcePosX",
# #         cs."sourcePosY",
# #         cs."sourcePosZ",
# #         cs.counts,
# #         cs.sipm_center_pos,
# #         lm.gamma_1_x,
# #         lm.gamma_1_y,
# #         lm.gamma_1_z,
# #         lm.gamma_2_x,
# #         lm.gamma_2_y,
# #         lm.gamma_2_z
# #     FROM
# #         coincidence_sample cs
# #         LEFT JOIN list_mode lm ON (cs. "eventID" = lm. "eventID");
# #     '''
# 
# import tensorflow as tf
# from hotpot.utils.tf_gpu import USEGPU
# 
# USEGPU(1)
# 
# sample_df = Database().read_sql(sample_stmt)
# 
# fa_count = FuncArray.from_pd_series(sample_df.counts)
# fa_sipm_center_pos = FuncArray.from_pd_series(sample_df.sipm_center_pos)
# 
# train_sample = (
#     fa_count.expand_dims(2)
#     .concatenate_with(fa_sipm_center_pos, 2)
#     .shrink((1, 2))
#     .rollaxis(1, 4)
#     .to_tensor()
# )
# 
# train_label = FuncArray(
#     sample_df[
#         ["gamma_1_x", "gamma_1_y", "gamma_1_z", "gamma_2_x", "gamma_2_y", "gamma_2_z"]
#     ]
# ).to_tensor()
# 
# print(train_sample.shape)
# print(train_label.shape)
# 
# train_ds = tf.data.Dataset.from_tensor_slices((train_sample, train_label))
# # train_ds = train_ds.take(5000).batch(32)
# train_ds = train_ds.batch(32)
# 
# tf.keras.backend.set_image_data_format("channels_last")
# 
# model = tf.keras.Sequential(
#     [
#         tf.keras.layers.Conv2D(32, 3, activation="relu"),
#         tf.keras.layers.Conv2D(64, 3, activation="relu"),
#         tf.keras.layers.Conv2D(128, 3, activation="relu"),
#         tf.keras.layers.Conv2D(64, 3, activation="relu"),
#         tf.keras.layers.Conv2D(32, 3, activation="relu"),
#         tf.keras.layers.Conv2D(16, 1, activation="relu"),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation="relu"),
#         tf.keras.layers.Dense(64, activation="relu"),
#         tf.keras.layers.Dense(32, activation="relu"),
#         tf.keras.layers.Dense(16, activation="relu"),
#         tf.keras.layers.Dense(6, activation="relu"),
#     ]
# )
# 
# 
# model.compile(
#     optimizer=tf.keras.optimizers.SGD(
#         learning_rate=0.000003, momentum=0.0, nesterov=False, name="SGD"
#     ),
#     loss="mse",
#     metrics=[tf.keras.metrics.MeanSquaredError()],
# )
# 
# 
# y = model(train_sample)
# 
# 
# model.fit(train_ds, batch_size=8, epochs=200)
