# import argsparse
# from hotpot.geometry.primiary import Cartesian3
# import tensorflow as tf
# 
# tf.keras.backend.set_image_data_format("channels_last")
# 
# import numpy as np
# import pandas as pd
# 
# import plotly.graph_objects as go
# 
# from hotpot.functools import FuncArray, FuncDataFrame, FuncNNLayer
# from hotpot.log import LossHistory
# from hotpot.loss import point_line_distance, point_line_distance_with_limitation
# 
# import os
# from hotpot.database import Database
# 
# os.environ[
#     "DB_CONNECTION"
# ] = "postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
# os.environ["PICLUSTER_DB"] = "postgresql://picluster@192.168.1.96:5432/picluster"
# 
# parser = argparse.ArgumentParser(description="Yo! Train something! Let's rock!")
# parser.add_argument('integers', metavar='train_cached_dataset_id', type=int, nargs='+', help='Train cached dataset id') 
# parser.add_argument('integers', metavar='valid_cached_dataset_id', type=int, nargs='+', help='Valid cached dataset id') 
# args = parser.parse_args()
# train_cached_dataset_id = args.train_cached_dataset_id
# valid_cached_dataset_id = args.valid_cached_dataset_id
# 
# 
# sipm_counts_n_position = CachedData(train_cached_dataset_id).sipm_counts_n_position
# anger_infered = CachedData(train_cached_dataset_id).anger_infered
# source_position = CachedData(train_cached_dataset_id).source_position
# 
# stacked_counts = tf.keras.Input(shape=(16, 16, 8), name="sipm_counts_n_position")
# anger = tf.keras.Input(shape=(6), name="anger_infered")
# 
# res_component = (
#     FuncNNLayer(stacked_counts)
#     .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
#     .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
# #     .append_next_layer(tf.keras.layers.Conv2D(32, 1, activation='relu', padding='same'))
#     .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
#     .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
#     .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
#     .append_next_layer(tf.keras.layers.Flatten())
#     .append_next_layer(tf.keras.layers.Dense(2048))
#     .append_next_layer(tf.keras.layers.Dropout(0.5))
#     .append_next_layer(tf.keras.layers.Dense(1024))
#     .append_next_layer(tf.keras.layers.Dense(1024))
# #     .append_next_layer(tf.keras.layers.Dropout(0.5))
#     .append_next_layer(tf.keras.layers.Dense(512))
#     .append_next_layer(tf.keras.layers.Dense(512))
# #     .append_next_layer(tf.keras.layers.Dropout(0.5))
#     .append_next_layer(tf.keras.layers.Dense(256))
#     .append_next_layer(tf.keras.layers.Dense(256))
# #     .append_next_layer(tf.keras.layers.Dropout(0.5))
#     .append_next_layer(tf.keras.layers.Dense(128))
#     .append_next_layer(tf.keras.layers.Dense(128))
# #     .append_next_layer(tf.keras.layers.Dropout(0.5))
#     .append_next_layer(tf.keras.layers.Dense(64))
#     .append_next_layer(tf.keras.layers.Dense(64))
# #     .append_next_layer(tf.keras.layers.Dropout(0.5))
#     .append_next_layer(tf.keras.layers.Dense(32))
#     .append_next_layer(tf.keras.layers.Dense(32))
#     .append_next_layer(tf.keras.layers.Dense(16))
#     .append_next_layer(tf.keras.layers.Dense(16))
#     .append_next_layer(tf.keras.layers.Dense(6))
# )
# 
# outputs = res_component.layer + anger
# 
# model = tf.keras.Model(inputs=[stacked_counts, anger], outputs=outputs)
# 
# model.compile(
#     optimizer=tf.keras.optimizers.SGD(
#         learning_rate=0.0001, momentum=0.1, nesterov=False, name="SGD"
#     ),
#     loss=point_line_distance_with_limitation,
# )
# history = LossHistory()
# 
# sipm_counts_n_position_valid = CachedData(8).sipm_counts_n_position
# anger_infered_valid = CachedData(8).anger_infered
# source_position_valid = CachedData(8).source_position
# 
# for i in range(100):
#     model.fit(
#         {"input_1": sipm_counts_n_position, "intput_2": anger_infered},
#         source_position,
#         batch_size=128,
#         epochs=100,
#         validation_split=0.1,
#         callbacks=[history],
#     )
#     model.save_weights(f"vars_train_on_{train_cached_dataset_id}_step_{i}.h5")
#     net_infered = model([sipm_counts_n_position_valid, anger_infered_valid])
#     with open(f"net_infered_{valid_cached_dataset_id}_step_{i}.npy", "wb") as f:
#         np.save(f, net_infered)
#     with open(f"loss_train_on_{train_cached_dataset_id}_step_{i}.npy", "wb") as f:
#         np.save(f, np.array(history.losses))
#     with open(f"val_loss_train_on_{train_cached_dataset_id}_step_{i}.npy", "wb") as f:
#         np.save(f, np.array(history.val_losses))
# 
# 
