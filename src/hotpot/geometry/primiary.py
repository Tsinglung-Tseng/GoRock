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

    @staticmethod
    def from_matrix(m):
        return Cartesian3(m[0], m[1], m[2])

    @staticmethod
    def from_tuple(TP):
        return Cartesian3(
                 x=tf.constant([TP[0]], dtype=tf.float64),
                 y=tf.constant([TP[1]], dtype=tf.float64),
                 z=tf.constant([TP[2]], dtype=tf.float64),
                )

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

    @staticmethod
    def from_tuple3s(tuple3s):
        t = np.array(tuple3s, dtype=np.float64).T
        return Cartesian3(
            x=t[0],
            y=t[1],
            z=t[2],
        )

    @classmethod
    def local_pos_from_hits(cls, hits):
        return Cartesian3(hits.localPosX, hits.localPosY, hits.localPosZ)

    @classmethod
    def pos_from_hits(cls, hits):
        return Cartesian3(hits.posX, hits.posY, hits.posZ)

    @classmethod
    def source_from_hits(cls, hits):
        return Cartesian3(hits.sourcePosX, hits.sourcePosY, hits.sourcePosZ)

    def move(self, by_vector):
        return Cartesian3(
            self.x + by_vector[0], self.y + by_vector[1], self.z + by_vector[2]
        )

    def rotate_ypr(self, rv_ypr):
        def rotation_matrix_x(angle):
            return tf.convert_to_tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(angle), -np.sin(angle)],
                    [0.0, np.sin(angle), np.cos(angle)],
                ]
            )

        def rotation_matrix_y(angle):
            return tf.convert_to_tensor(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            )

        def rotation_matrix_z(angle):
            return tf.convert_to_tensor(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

        return (
            self.left_matmul(rotation_matrix_x(rv_ypr[0]))
            .left_matmul(rotation_matrix_y(rv_ypr[1]))
            .left_matmul(rotation_matrix_z(rv_ypr[2]))
        )

    def left_matmul(self, m):
        result = np.matmul(m, self.to_matrix())
        return Cartesian3.from_matrix(result)

    def distance_to(self, other):
        diff = self - other
        return tf.sqrt(tf.reduce_sum(tf.square(diff.to_tensor()), axis=0))

    def to_matrix(self):
        return tf.stack([self.x, self.y, self.z], axis=0)

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

    def to_plotly(self, mode="markers", **kwargs):
        return go.Scatter3d(
            x=self.x,
            y=self.y,
            z=self.z,
            mode=mode,
            **kwargs
            # marker=marker
        )

    def to_plotly_as_mesh3d(self, **marker):
        return go.Mesh3d(
            x=self.x,
            y=self.y,
            z=self.z,
            marker=marker
        )

    # def to_ipyvolume(self):
        # fig = ipv.figure()
        # scatter = ipv.scatter(self.x, self.y, self.z, size=1, marker="sphere")
        # return ipv
        # ipv.show()
        # ipv.quickscatter(self.x, self.y, self.z, ")


class Surface:
    def __init__(self, vertices: Cartesian3):
        self.vertices = vertices

    @staticmethod
    def from_xy_size(x_length, y_length):
        vertex_x = tf.convert_to_tensor(
            [x_length / 2, x_length / 2, -x_length / 2, -x_length / 2], dtype=tf.float64
        )
        vertex_y = tf.convert_to_tensor(
            [y_length / 2, -y_length / 2, y_length / 2, -y_length / 2], dtype=tf.float64
        )
        vertex_z = tf.convert_to_tensor([0.0, 0.0, 0.0, 0.0], dtype=tf.float64)
        return Surface(Cartesian3(vertex_x, vertex_y, vertex_z))

    def move(self, move_vector):
        return Surface(self.vertices + Cartesian3.from_tuple(move_vector))

    def rotate_ypr(self, rv_ypr):
        return Surface(self.vertices.rotate_ypr(rv_ypr))

    def to_plotly(self, **marker):
        return go.Mesh3d(
            x = self.vertices.x,
            y = self.vertices.y,
            z = self.vertices.z,
            i=[0,1],
            j=[1,2],
            k=[2,3],
            opacity=0.2,
            color='lightblue',
            **marker
        )


class Vector(Cartesian3):
    def __init__(self, x, y, z):
        super().__init__(
            tf.convert_to_tensor(x, dtype=tf.float64), 
            tf.convert_to_tensor(y, dtype=tf.float64), 
            tf.convert_to_tensor(z, dtype=tf.float64)
        )
        
    def dot(self, other):
        return Vector(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z
        )
    
    def cross(self, other):
        return Vector(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )
    
    @property
    def norm(self):
        return tf.sqrt(tf.reduce_sum([tf.square(self.x), tf.square(self.y), tf.square(self.z)], axis=0))
    
    def unit(self):
        return Vector(
            self.x / self.norm,
            self.y / self.norm,
            self.z / self.norm,
        )


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
    def __init__(self, surface):
        self.surface = surface
        
    @staticmethod
    def from_size(x, y, z):
        return Box([
            Surface.from_xy_size(x, y).move([0, 0, z/2]),
            Surface.from_xy_size(x, y).move([0, 0, -z/2]),
            Surface.from_xy_size(x, z).rotate_ypr([np.pi/2, 0, 0]).move([0, y/2, 0]),
            Surface.from_xy_size(x, z).rotate_ypr([np.pi/2, 0, 0]).move([0, -y/2, 0]),
            Surface.from_xy_size(y, z).rotate_ypr([0, 0, np.pi/2]).rotate_ypr([0, np.pi/2, 0]).move([x/2, 0, 0]),
            Surface.from_xy_size(y, z).rotate_ypr([0, 0, np.pi/2]).rotate_ypr([0, np.pi/2, 0]).move([-x/2, 0, 0]),
        ]) 
          
    @staticmethod
    def from_vertices_of_surface(vertices):
        return Box([
            Surface(Cartesian3(x, y, z))
            for x, y, z in
            zip(
            tf.split(vertices.x, 6, axis=0), 
            tf.split(vertices.y, 6, axis=0), 
            tf.split(vertices.z, 6, axis=0)
        )])
    
    @property
    def vertices(self):
        return Cartesian3.from_matrix(tf.concat([s.vertices.to_tensor() for s in self.surface], axis=1))
    
    def to_plotly(self):
        return [
            s.to_plotly()
            for s in self.surface
        ]
    
    def move(self, move_vector):
        return Box.from_vertices_of_surface(self.vertices + Cartesian3.from_tuple(move_vector))

    def rotate_ypr(self, rv_ypr):
        return Box.from_vertices_of_surface(self.vertices.rotate_ypr(rv_ypr))

