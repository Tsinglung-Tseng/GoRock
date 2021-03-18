import plotly.graph_objects as go
import numpy as np
import sympy as sp
import operator
import functools

# import ipyvolume as ipv
from ..functools import FuncArray
import tensorflow as tf
import matplotlib.pyplot as plt
from functools import partial, reduce
from collections.abc import Iterable
from ..database import Database


def split_raw_df_into_even_odd_pairs(raw_df):
    return (raw_df[0::2], raw_df[1::2])


def _convert_type_if_not_nparray(an_array):
    if not isinstance(an_array, np.ndarray):
        return np.array(an_array, dtype=np.float32)
    else:
        return an_array


class Cartesian3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def fmap(self, func):
        return self.__class__(x=func(self.x), y=func(self.y), z=func(self.z))

    def hmap(self, func):
        """
        Map function horizontally on each element of param arrays.
        """
        return reduce(func, [self.x, self.y, self.z])

    def op_zip(self, other, op):
        return self.__class__.from_xyz(
            x=op(self.x, other.x), y=op(self.y, other.y), z=op(self.z, other.z)
        )

    @staticmethod
    def from_xyz(x, y, z):
        return Cartesian3(x, y, z).fmap(_convert_type_if_not_nparray)

    @staticmethod
    def from_tuple(t):
        return Cartesian3(*t)

    @staticmethod
    def from_tuple3s(TPs):
        TPs = _convert_type_if_not_nparray(TPs)
        return Cartesian3.from_xyz(x=TPs[:, 0], y=TPs[:, 1], z=TPs[:, 2])

    @staticmethod
    def from_matrix(m):
        return Cartesian3.from_xyz(x=m[0], y=m[1], z=m[2])

    @staticmethod
    def from_cartesian3s(list_of_carteisan3):
        return reduce(lambda car, cdr: car.concat(cdr), list_of_carteisan3)

    @classmethod
    def local_pos_from_hits(cls, hits):
        return Cartesian3.from_xyz(hits.localPosX, hits.localPosY, hits.localPosZ)

    @classmethod
    def pos_from_hits(cls, hits):
        return Cartesian3.from_xyz(hits.posX, hits.posY, hits.posZ)

    @classmethod
    def source_from_hits(cls, hits):
        return Cartesian3.from_xyz(hits.sourcePosX, hits.sourcePosY, hits.sourcePosZ)

    def __repr__(self):
        try:
            return f"""{self.__class__.__name__} <size: ({len(self.x)}, {len(self.y)}, {len(self.z)}), x: {self.x}, y: {self.y}, z: {self.z}>"""
        except:
            return f"""{self.__class__.__name__} <size: (1, 1, 1), x: {self.x}, y: {self.y}, z: {self.z}>"""

    def __getitem__(self, idx):
        return Cartesian3.from_xyz(
            self.x[idx],
            self.y[idx],
            self.z[idx],
        )

    def __add__(self, other):
        return self.op_zip(other, np.add)

    def __sub__(self, other):
        return self.op_zip(other, np.subtract)

    def __mul__(self, op_num):
        mul_num = lambda x: np.multiply(x, op_num)
        return self.fmap(mul_num)

    def __truediv__(self, other):
        return Cartesian3(self.x / other, self.y / other, self.z / other)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        def _getter(item):
            if isinstance(item[key], Iterable):
                return item[key]
            else:
                return np.array([item[key]])

        return self.fmap(_getter)

    @property
    def shape(self):
        return np.array([len(self), 3])

    def divide(self, other):
        return self.op_zip(other, np.divide)

    def move(self, by_vector):
        return Cartesian3(
            self.x + by_vector[0], self.y + by_vector[1], self.z + by_vector[2]
        )

    def concat(self, other):
        def _np_concat():
            return partial(np.concatenate, axis=0)

        return self.__class__.from_xyz(
            x=_np_concat()([self.x, other.x]),
            y=_np_concat()([self.y, other.y]),
            z=_np_concat()([self.z, other.z]),
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

    def close_enough_to(self, other):
        return (
            (self - other)
            .fmap(lambda i: abs(i) < 0.001)
            .fmap(
                lambda i: functools.reduce(lambda car, cdr: operator.and_(car, cdr), i)
            )
            .hmap(lambda car, cdr: operator.and_(car, cdr))
        )

    def distance_to(self, other):
        diff = self - other
        return tf.sqrt(tf.reduce_sum(tf.square(diff.to_tensor()), axis=0))

    def to_matrix(self):
        return tf.stack([self.x, self.y, self.z], axis=0)

    def to_numpy(self):
        return self.to_tensor().numpy()

    def to_list(self):
        return self.to_tensor().numpy().tolist()

    def to_func_array(self):
        return FuncArray(self.to_numpy())

    def to_tensor(self):
        return tf.stack([tf.constant(self.x), tf.constant(self.y), tf.constant(self.z)])

    def to_sp_point3d(self):
        return [
            sp.Point3D(
                p.x[0],
                p.y[0],
                p.z[0]
            )
            for p in self
        ]

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
        return go.Scatter3d(x=self.x, y=self.y, z=self.z, mode=mode, **kwargs)

    # def to_plotly_as_surface(self):
    # np
    # return go.Surface(z=self.)

    def to_plotly_as_mesh3d(self, **kwargs):
        return go.Mesh3d(x=self.x, y=self.y, z=self.z, **kwargs)

    def view(self, figsize=(30, 10)):
        plt.figure(figsize=figsize)
        x_view = plt.subplot(131, aspect="equal")
        x_view.plot(self.y, self.z, ".")
        x_view.set_title("x_view")

        y_view = plt.subplot(132, aspect="equal", sharex=x_view)
        y_view.plot(self.x, self.z, ".")
        y_view.set_title("y_view")

        z_view = plt.subplot(133, aspect="equal", sharex=x_view)
        z_view.plot(self.x, self.y, ".")
        z_view.set_title("z_view")

        plt.show()


class Segment:
    """
    hits = pd.read_csv('/home/zengqinglong/optical_simu/system_Albira_3_ring_debug/hits_box.csv')
    hits_pos = Cartesian3.from_tuple3s(hits[['posX', 'posY', 'posZ']].to_numpy())
    s_10 = Segment(hits_pos[:10], hits_pos[10:20])
    go.Figure([
        *s_10.to_plotly_line(),
        *s_10.to_plotly_segment(mode='lines+markers', marker=dict(size=3))
    ])
    """

    def __init__(self, fst: Cartesian3, snd: Cartesian3):
        self.fst = fst
        self.snd = snd

    @staticmethod
    def from_listmode(lm):
        return Segment(
            fst=Cartesian3.from_tuple3s(lm[:, :3]),
            snd=Cartesian3.from_tuple3s(lm[:, 3:]),
        )

    def __repr__(self):
        return f"""Pair: <fst: {self.fst}; snd: {self.snd}>"""

    def __getitem__(self, key):
        return self.fmap(lambda i: i[key])

    def fmap(self, func):
        return self.__class__(fst=func(self.fst), snd=func(self.snd))

    def seg_length(self):
        return self.fst.distance_to(self.snd)

    def direct_vector(self):
        return (self.fst - self.snd).fmap(lambda i: i / self.seg_length().numpy())

    @property
    def middle_point(self):
        return (self.fst + self.snd) / 2

    def to_listmode(self):
        return np.hstack([self.fst.to_numpy().T, self.snd.to_numpy().T])
        # return np.stack([self.fst.to_matrix(), self.snd.to_matrix()], axis=0)

    def to_sp_line3d(self):
        return FuncArray([
            sp.Line3D(segment.fst.to_sp_point3d()[0], segment.snd.to_sp_point3d()[0])
            for segment in self
        ])

    def to_plotly_line(
        self, line_length=600, mode="lines", marker=dict(size=3), **kvargs
    ):
        lines = Segment(
            self.middle_point + self.direct_vector() * line_length / 2,
            self.middle_point - self.direct_vector() * line_length / 2,
        )
        return lines.to_plotly_segment(mode=mode, marker=marker, **kvargs)

    def to_plotly_segment(self, mode="markers+lines", marker=dict(size=3), **kwargs):
        tmp = []
        for i in self:
            tmp.append(i.fst.concat(i.snd).to_plotly(mode, marker=marker, **kwargs))
        return tmp


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
            x=self.vertices.x,
            y=self.vertices.y,
            z=self.vertices.z,
            i=[0, 1],
            j=[1, 2],
            k=[2, 3],
            opacity=0.2,
            # color='rgba(255,0,255, 0.4)',
            color="lightblue",
            **marker,
        )


class Vector(Cartesian3):
    def __init__(self, x, y, z):
        super().__init__(
            tf.convert_to_tensor(x, dtype=tf.float64),
            tf.convert_to_tensor(y, dtype=tf.float64),
            tf.convert_to_tensor(z, dtype=tf.float64),
        )

    def dot(self, other):
        return Vector(self.x * other.x, self.y * other.y, self.z * other.z)

    def cross(self, other):
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    @property
    def norm(self):
        return tf.sqrt(
            tf.reduce_sum(
                [tf.square(self.x), tf.square(self.y), tf.square(self.z)], axis=0
            )
        )

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
    def __init__(self, fst: Cartesian3, snd: Cartesian3):
        """
        sample_df = Database().read_sql(sample_stmt)

        gamma_1 = Cartesian3.from_tuple3s(FuncDataFrame(sample_df).select(['gamma_1_x','gamma_1_y','gamma_1_z']).to_numpy())
        gamma_2 = Cartesian3.from_tuple3s(FuncDataFrame(sample_df).select(['gamma_2_x','gamma_2_y','gamma_2_z']).to_numpy())

        gamma_pair = Segment(gamma_1, gamma_2)

        go.Figure([
            *gamma_pair.to_plotly_line(line_length=800, marker=dict(size=5, color='gold')),
            gamma_pair.middle_point.to_plotly(mode="markers", marker=dict(size=2, color='purple')),
            gamma_1.to_plotly(mode="markers", marker=dict(size=5, color='red')),
            gamma_2.to_plotly(mode="markers", marker=dict(size=5, color='blue')),
        ])
        """
        self.fst = fst
        self.snd = snd

    def __repr__(self):
        return f"""Pair: <fst: {self.fst}; snd: {self.snd}>"""

    def hmap(self):
        pass

    def seg_length(self):
        return self.fst.distance_to(self.snd)

    def direct_vector(self):
        return (self.fst - self.snd).fmap(lambda i: i / self.seg_length().numpy())

    @property
    def middle_point(self):
        return (self.fst + self.snd) / 2

    def to_listmode(self):
        return np.stack([self.fst.to_matrix(), self.snd.to_matrix()], axis=0)

    def to_plotly_line(self, line_length=600, mode="lines", **kwargs):
        tmp = []

        lm = Pair(
            self.middle_point + self.direct_vector() * line_length / 2,
            self.middle_point - self.direct_vector() * line_length / 2,
        ).to_listmode()

        for i in range(lm.shape[2]):
            x, y, z = lm[:, :, i].T
            tmp.append(go.Scatter3d(x=x, y=y, z=z, mode=mode, **kwargs))
        return tmp

    def to_plotly_segment(self, mode="markers+lines", **kwargs):
        tmp = []

        lm = gamma_pair.to_listmode()
        for i in range(lm.shape[2]):
            x, y, z = lm[:, :, i].T
            tmp.append(go.Scatter3d(x=x, y=y, z=z, mode=mode, **kwargs))
        return tmp


class Box:
    def __init__(self, surface):
        self.surface = surface

    @staticmethod
    def from_size(x, y, z):
        return Box(
            [
                Surface.from_xy_size(x, y).move([0, 0, z / 2]),
                Surface.from_xy_size(x, y).move([0, 0, -z / 2]),
                Surface.from_xy_size(x, z)
                .rotate_ypr([np.pi / 2, 0, 0])
                .move([0, y / 2, 0]),
                Surface.from_xy_size(x, z)
                .rotate_ypr([np.pi / 2, 0, 0])
                .move([0, -y / 2, 0]),
                Surface.from_xy_size(y, z)
                .rotate_ypr([0, 0, np.pi / 2])
                .rotate_ypr([0, np.pi / 2, 0])
                .move([x / 2, 0, 0]),
                Surface.from_xy_size(y, z)
                .rotate_ypr([0, 0, np.pi / 2])
                .rotate_ypr([0, np.pi / 2, 0])
                .move([-x / 2, 0, 0]),
            ]
        )

    @staticmethod
    def from_vertices_of_surface(vertices):
        return Box(
            [
                Surface(Cartesian3(x, y, z))
                for x, y, z in zip(
                    tf.split(vertices.x, 6, axis=0),
                    tf.split(vertices.y, 6, axis=0),
                    tf.split(vertices.z, 6, axis=0),
                )
            ]
        )

    @property
    def vertices(self):
        return Cartesian3.from_matrix(
            tf.concat([s.vertices.to_tensor() for s in self.surface], axis=1)
        )

    def to_plotly(self):
        return [s.to_plotly() for s in self.surface]

    def move(self, move_vector):
        return Box.from_vertices_of_surface(
            self.vertices + Cartesian3.from_tuple(move_vector)
        )

    def rotate_ypr(self, rv_ypr):
        return Box.from_vertices_of_surface(self.vertices.rotate_ypr(rv_ypr))


class Trapezoid:
    def __init__(self, vertices: Cartesian3):
        self.vertices = vertices

    @staticmethod
    def from_size(lower_side_length, higher_side_length, hight):
        square_surface = lambda side_lenght: Cartesian3.from_tuple3s(
            [
                [side_lenght / 2, side_lenght / 2, 0],
                [side_lenght / 2, -side_lenght / 2, 0],
                [-side_lenght / 2, side_lenght / 2, 0],
                [-side_lenght / 2, -side_lenght / 2, 0],
            ]
        )

        side_surface = lambda i, j: lower_surface_vertices.fmap(
            lambda x: [x[i], x[j]]
        ).concat(higher_surface_vertices.fmap(lambda x: [x[i], x[j]]))

        lower_surface_vertices = square_surface(lower_side_length).move(
            [0, 0, -hight / 2]
        )
        higher_surface_vertices = square_surface(higher_side_length).move(
            [0, 0, hight / 2]
        )

        side_1_vertices = side_surface(0, 1)
        side_2_vertices = side_surface(0, 2)
        side_3_vertices = side_surface(2, 3)
        side_4_vertices = side_surface(1, 3)

        return Trapezoid(
            Cartesian3.from_cartesian3s(
                [
                    lower_surface_vertices,
                    higher_surface_vertices,
                    side_1_vertices,
                    side_2_vertices,
                    side_3_vertices,
                    side_4_vertices,
                ]
            )
        )

    def to_surface_vertices(self):
        split_index = lambda length, split_by: [
            np.arange(*r)
            for r in list(
                zip(
                    np.arange(0, length, split_by),
                    np.arange(0, length, split_by) + split_by,
                )
            )
        ]

        return [self.vertices[r] for r in split_index(24, 4)]

    def to_plotly(self):
        return [Surface(s).to_plotly() for s in self.to_surface_vertices()]
