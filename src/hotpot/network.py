import tensorflow as tf
from .loss import point_line_distance, point_line_distance_with_limitation


tf.keras.backend.set_image_data_format('channels_last')

stacked_counts = tf.keras.Input(shape=(16,16,8), name='sipm_counts_n_position')
anger = tf.keras.Input(shape=(6), name='anger_infered')

conv_1 = tf.keras.layers.Conv2D(32, 3, activation='relu')(stacked_counts)
conv_2 = tf.keras.layers.Conv2D(64, 3, activation='relu')(conv_1)
conv_3 = tf.keras.layers.Conv2D(128, 3, activation='relu')(conv_2)
conv_4 = tf.keras.layers.Conv2D(64, 3, activation='relu')(conv_3)
conv_5 = tf.keras.layers.Conv2D(32, 3, activation='relu')(conv_4)
flatten_conv = tf.keras.layers.Flatten()(conv_5)
dense_1 = tf.keras.layers.Dense(128, activation='relu')(flatten_conv)
dense_2 = tf.keras.layers.Dense(64, activation='relu')(dense_1)
dense_3 = tf.keras.layers.Dense(32, activation='relu')(dense_2)
dense_4 = tf.keras.layers.Dense(16, activation='relu')(dense_3)
dense_5 = tf.keras.layers.Dense(6, activation='relu')(dense_4)

outputs = tf.keras.layers.add([dense_5, anger])
model_res = tf.keras.Model(inputs=[stacked_counts, anger], outputs=outputs)

model_res.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.0001, momentum=0.1, nesterov=False, name='SGD'
    ),
    loss=point_line_distance,
#     metrics=[tf.keras.metrics.MeanAbsoluteError()]
)



model_res_limit_loss = tf.keras.Model(inputs=[stacked_counts, anger], outputs=outputs)

model_res_limit_loss.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.001, momentum=0.1, nesterov=False, name='SGD'
    ),
    loss=point_line_distance_with_limitation,
#     metrics=[tf.keras.metrics.MeanAbsoluteError()]
)
