from hotpot.functools import FuncArray, FuncDataFrame, FuncNNLayer
import tensorflow as tf


def point_to_point_loss(x, y):
    return tf.math.sqrt(tf.math.reduce_sum(tf.square(x-y), axis=1))


counts = tf.keras.Input(shape=(16,16,1), name="counts")
interaction_pos = tf.keras.Input(shape=(3), name="interaction_pos")

model = (
    FuncNNLayer(counts)
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'))
    .append_next_layer(tf.keras.layers.Flatten())
    .append_next_layer(tf.keras.layers.Dense(1024))
    .append_next_layer(tf.keras.layers.Dropout(0.5))
    .append_next_layer(tf.keras.layers.Dense(512))
    .append_next_layer(tf.keras.layers.Dense(256))
    .append_next_layer(tf.keras.layers.Dense(128))
    .append_next_layer(tf.keras.layers.Dense(64))
    .append_next_layer(tf.keras.layers.Dense(32))
    .append_next_layer(tf.keras.layers.Dense(16))
    .append_next_layer(tf.keras.layers.Dense(8))
    .append_next_layer(tf.keras.layers.Dense(3))
)

outputs=model.layer

model = tf.keras.Model(inputs=counts, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.1, nesterov=False, name="SGD"),
    loss=point_to_point_loss#tf.keras.metrics.MAE,
)

