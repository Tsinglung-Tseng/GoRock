import tensorflow as tf


def point_line_distance(y_true, y_pred):
    
    source = y_true
    A = y_pred[:,3:]
    B = y_pred[:,:3]

    mag = lambda x: tf.sqrt(tf.math.reduce_sum(x*x, axis=-1))
    return mag(tf.linalg.cross((B-A), (source-A)))/mag(B-A)


def point_line_distance_with_limitation(y_true, y_pred):

    def _is_excedes_sys(p):
        rrr = tf.square(p[:,0]) + tf.square(p[:,2])
        return rrr<44100 | rrr>50625 | p[:,0]>115 | p[:,0]<-115

    source = y_true
    A = y_pred[:,3:]
    B = y_pred[:,:3]

    if _is_excedes_sys(A) | _is_excedes_sys(B):
        return 1000

    mag = lambda x: tf.sqrt(tf.math.reduce_sum(x*x, axis=-1))
    return mag(tf.linalg.cross((B-A), (source-A)))/mag(B-A)


def point_line_distance_with_limitation(y_true, y_pred):

    source = y_true
    A = y_pred[:,3:]
    B = y_pred[:,:3]

    mask_1 = tf.square(A[:,0]) + tf.square(A[:,2])<44100
    mask_2 = tf.square(A[:,0]) + tf.square(A[:,2])>50625
    mask_3 = tf.square(B[:,0]) + tf.square(B[:,2])<44100
    mask_4 = tf.square(B[:,0]) + tf.square(B[:,2])>50625
    full_mask = tf.logical_not(tf.logical_or(tf.logical_or(tf.logical_or(mask_1, mask_2), mask_3), mask_4))

    A = tf.boolean_mask(A, full_mask)
    B = tf.boolean_mask(B, full_mask)
    source = tf.boolean_mask(source, full_mask)

#     print(f"{A.shape}, {B.shape}, {source.shape}")

#     return A,B,source
    mag = lambda x: tf.sqrt(tf.math.reduce_sum(x*x, axis=-1))
    return mag(tf.linalg.cross((B-A), (source-A)))/mag(B-A)
