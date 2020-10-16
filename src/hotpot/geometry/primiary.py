import plotly.graph_objects as go
import numpy as np
import ipyvolume as ipv
import tensorflow as tf
from ..database import Database

# def point_close_enough(c: Cartisian3):
#     return


# def simple_pair_similar_enough(x, y):
#     return list(map(lambda x: x < 0.00001, np.array([i - j for i, j in zip(x, y)])))


# def shift_loop_array(x):
#     pass


def split_raw_df_into_even_odd_pairs(raw_df):
    return (raw_df[0::2], raw_df[1::2])


class Cartesian3:
    def __init__(self, x, y, z):
        self.x = tf.constant(x)
        self.y = tf.constant(y)
        self.z = tf.constant(z)

    def fmap(self, f):
        return Cartesian3(f(self.x), f(self.y), f(self.z))

    def zip_op(self, op, other):
        return Cartesian3(op(self.x, other.x), op(self.y, other.y), op(self.z, other.z))

    def __repr__(self):
        return f"""{self.__class__.__name__} <size: ({len(self.x)}, {len(self.y)}, {len(self.z)}), x: {self.x}, y: {self.y}, z: {self.z}>"""

    def __add__(self, other):
        return Cartesian3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Cartesian3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __truediv__(self, other):
        return Cartesian3(self.x / other, self.y / other, self.z / other)

    @classmethod
    def from_pattern(cls, raw, pattern):
        return Cartesian3(
            x=raw[pattern + "x"], y=raw[pattern + "y"], z=raw[pattern + "z"]
        )

    @classmethod
    def local_pos_from_hits(cls, hits):
        return Cartesian3(hits.localPosX, hits.localPosY, hits.localPosZ)

    @classmethod
    def pos_from_hits(cls, hits):
        return Cartesian3(hits.posX, hits.posY, hits.posZ)

    def move(self, by_vector):
        return Cartesian3(
            self.x + by_vector[0], self.y + by_vector[1], self.z + by_vector[2]
        )

    def rotate_using_rotate_matrix(self, rotate_matrix):
        rotated = tf.linalg.matmul(rotate_matrix, self.to_tensor())
        rotatedX = rotated.numpy()[0, :]
        rotatedY = rotated.numpy()[1, :]
        rotatedZ = rotated.numpy()[2, :]
        return Cartesian3(rotatedX, rotatedY, rotatedZ)

    def distance_to(self, other):
        diff = self - other
        return tf.sqrt(tf.reduce_sum(tf.square(diff.to_tensor()), axis=0))

    def to_tensor(self):
        return tf.stack([tf.constant(self.x), tf.constant(self.y), tf.constant(self.z)])

    def to_spherical(self):
        return Spherical(
            r=tf.math.sqrt(
                sum([tf.square(self.x), tf.square(self.y), tf.square(self.z)])
            ),
            theta=tf.math.atan2(self.y, self.x),
            phi=tf.math.atan2(
                tf.math.sqrt(tf.square(self.x) + tf.square(self.y)), self.z
            ),
        )

    def to_plotly(self):
        return go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode="markers",
            marker=dict(
                size=1,
                #                 color=z,                # set color to an array/list of desired values
                #                 colorscale='Viridis',   # choose a colorscale
                #                 opacity=0.8
            ),
        )

    def to_ipyvolume(self):
        fig = ipv.figure()
        scatter = ipv.scatter(self.x, self.y, self.z, size=1, marker="sphere")
        return ipv
        # ipv.show()
        # ipv.quickscatter(self.x, self.y, self.z, ")


class Spherical:
    def __init__(self, r, theta, phi):
        self.r = r
        self.theta = theta
        self.phi = phi

    def to_cartesian(self):
        return Cartesian3(
            x=self.r * tf.math.sin(self.phi) * tf.math.cos(self.theta),
            y=self.r * tf.math.sin(self.phi) * tf.math.sin(self.theta),
            z=self.r * tf.math.cos(self.phi),
        )


def get_source(samples):
    return Cartesian3(
        np.array(samples["source_x"]),
        np.array(samples["source_y"]),
        np.array(samples["source_z"]),
    )


class Pair:
    def __init__(self, fst, snd):
        self.fst = fst
        self.snd = snd


class PairCartesian3(Pair, Cartesian3):
    def __init__(self, raw, pattern):
        source = Cartesian3.from_pattern(raw, pattern)
        Cartesian3.__init__(self, source.x, source.y, source.z)

        fst, snd = split_raw_df_into_even_odd_pairs(raw)
        Pair.__init__(
            self,
            Cartesian3.from_pattern(fst, pattern),
            Cartesian3.from_pattern(snd, pattern),
        )


class Segment(Pair):
    def __init__(self, car: Cartesian3, cdr: Cartesian3):
        self.car = car
        self.cdr = cdr

    def to_plotly(self):
        return [
            go.Scatter3d(x=[x0, x1], y=[y0, y1], z=[z0, z1], mode="lines")
            for x0, x1, y0, y1, z0, z1 in zip(
                np.array(self.car.x).flat,
                np.array(self.cdr.x).flat,
                np.array(self.car.y).flat,
                np.array(self.cdr.y).flat,
                np.array(self.car.z).flat,
                np.array(self.cdr.z).flat,
            )
        ]


class Box:
    def __init__(self, size_x, size_y, size_z):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z

    @property
    def vertex(self):
        x = [
            self.size_x / 2,
            self.size_x / 2,
            self.size_x / 2,
            self.size_x / 2,
            -self.size_x / 2,
            -self.size_x / 2,
            -self.size_x / 2,
            -self.size_x / 2,
        ]
        y = [
            self.size_y / 2,
            self.size_y / 2,
            -self.size_y / 2,
            -self.size_y / 2,
            self.size_y / 2,
            self.size_y / 2,
            -self.size_y / 2,
            -self.size_y / 2,
        ]
        z = [
            self.size_z / 2,
            -self.size_z / 2,
            -self.size_z / 2,
            self.size_z / 2,
            self.size_z / 2,
            -self.size_z / 2,
            -self.size_z / 2,
            self.size_z / 2,
        ]
        return Cartesian3(x, y, z)

    def to_plotly(self):
        return go.Mesh3d(
            x=self.vertex.x,
            y=self.vertex.y,
            z=self.vertex.z,
            color="rgb(141,160,203)",
            opacity=0.5,
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            showscale=True,
        )
