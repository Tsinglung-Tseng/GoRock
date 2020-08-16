import tensorflow as tf
import plotly.graph_objects as go
import numpy as np

# for mass center
INDICES = tf.range(16, dtype=tf.float32)
ROW_AXIS = 1
COLUMN_AXIS = 0
SIPM_GAP = 3.26


def batch_unify(batch_vectors):
    batch_magnitude = tf.math.sqrt(tf.einsum("ijk->ik", tf.math.square(batch_vectors)))
    unit_batch_vectors = tf.einsum(
        "ijk,ik->ijk", batch_vectors, tf.math.reciprocal(batch_magnitude)
    )
    return unit_batch_vectors


def batch_magnitude(bv):
    return tf.math.sqrt(tf.einsum("ijk->ik", tf.math.square(bv)))


def batch_cross(bv_1, bv_2):
    LCSymbol3 = tf.constant(
        [
            [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        ],
        dtype=tf.float32,
    )
    return tf.einsum("ijk,ljn,lkn->lin", LCSymbol3, bv_1, bv_2)


def batch_anger(sample):
    def X_c(n_CR):
        E_R_n_CR = tf.math.reduce_sum(n_CR, axis=2)
        return tf.math.reduce_sum(
            tf.multiply(E_R_n_CR, INDICES), axis=1
        ) / tf.math.reduce_sum(E_R_n_CR, axis=1)

    def Y_c(n_CR):
        E_R_n_CR = tf.math.reduce_sum(n_CR, axis=1)
        return tf.math.reduce_sum(
            tf.multiply(E_R_n_CR, INDICES), axis=1
        ) / tf.math.reduce_sum(E_R_n_CR, axis=1)

    index_0 = tf.stack(
        [X_c(sample.count[:, :, :, 0]), Y_c(sample.count[:, :, :, 0])], axis=1
    )
    index_1 = tf.stack(
        [X_c(sample.count[:, :, :, 1]), Y_c(sample.count[:, :, :, 1])], axis=1
    )
    return tf.stack([index_0, index_1], axis=2)


def batch_cart_to_plotly(batch_cart3):
    return go.Scatter3d(
        {
            k: np.reshape(v, np.prod(v.shape))
            for k, v in zip(
                ["x", "y", "z"], np.split(batch_cart3.numpy(), [1, 2, 3], axis=1)
            )
        },
        mode='markers',
    )


def sipm_local_index_to_world_coordinate(sample):

    def get_sipm_center_at(sample, x, y):
        return tf.stack(
            [
                sample.centers.x[:, x, y, :],
                sample.centers.y[:, x, y, :],
                sample.centers.z[:, x, y, :],
            ],
            axis=1,
        )

    e0 = get_sipm_center_at(sample, 0, 0)
    # SiPM gap on x/y direction respectively
    ex = batch_unify(
        get_sipm_center_at(sample, 1, 0) - get_sipm_center_at(sample, 0, 0)
    )
    ey = batch_unify(
        get_sipm_center_at(sample, 0, 1) - get_sipm_center_at(sample, 0, 0)
    )
    ez = batch_cross(ex, ey)

    # TODO: use mess object to replate batch_anger
    batch_anger_index = batch_anger(sample)

    crystal_local_base_matrix = tf.stack([ex, ey, ez], axis=1)

    sipm_local_coor = tf.concat(
        [batch_anger_index * SIPM_GAP, tf.zeros([sample.to_label().shape[0], 1, 2])],
        axis=1,
    )
    sipm_global_coor = (
        tf.einsum("ijkl,ijl->ikl", crystal_local_base_matrix, sipm_local_coor) + e0
    )

    return sipm_global_coor
