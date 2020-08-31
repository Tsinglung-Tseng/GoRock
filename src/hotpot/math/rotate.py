import numpy as np
import tensorflow as tf
from ..geometry.primiary import Cartisian3


class Vector3(Cartisian3):
    def __init__(self, x, y, z):
        super().__init__(x, y, z)
        # self.x = tf.constant(x)
        # self.y = tf.constant(y)
        # self.z = tf.constant(z)

    def __mul__(self, that):
        return self.__class__(
            self.x * that,
            self.y * that,
            self.z * that,
        )

        
    # def to_numpy(self):
    #     return (np.array(self.x), np.array(self.y))


class UnitVector:
    def __init__(self, theta, phi):
        self.theta = tf.constant(theta)
        self.phi = tf.constant(phi)
        
    def to_vector3(self):
        return Vector3(
            x = tf.sin(self.theta)*tf.cos(self.theta),
            y = tf.sin(self.theta)*tf.sin(self.phi),
            z = tf.cos(self.theta)
        )


class Quaternion:
    def __init__(self, a, b, c, d):
        self.a = tf.constant(a)
        self.b = tf.constant(b)
        self.c = tf.constant(c)
        self.d = tf.constant(d)
    
    def __add__(self, that):
        return Quaternion(
            self.a+that.a,
            self.b+that.a,
            self.c+that.c,
            self.d+that.d,
        )
    
    def __sub__(self, that):
        return Quaternion(
            self.a-that.a,
            self.b-that.a,
            self.c-that.c,
            self.d-that.d,
        )
    
    def __mul__(self, that):
        return Quaternion(
            self.a*that.a - self.b*that.b - self.c*that.c - self.d*that.d,
            self.b*that.a + self.a*that.b + self.d*that.c - self.c*that.d,
            self.c*that.a + self.d*that.b + self.a*that.c - self.b*that.d,
            self.d*that.a + self.c*that.b + self.b*that.c - self.a*that.d,
        )
    
    def __repr__(self):
        return f"""<Quaternion: a={self.a}, b={self.b}, c={self.c}, d={self.d}>"""
    
    @classmethod
    def from_axis_angle(cls, axis: UnitVector, angle):
        ijk = axis.to_vector3()*tf.sin(angle/2)
        return Quaternion(
            a = tf.cos(angle/2),
            b = ijk.x,
            c = ijk.y,
            d = ijk.z
        )

    def to_rotation_matrix(self):
        return tf.stack([
            [1 - 2*tf.square(self.c) - 2*tf.square(self.d), 2*self.b*self.c-2*self.a*self.d, 2*self.a*self.c + 2*self.b*self.d],
            [2*self.b*self.c + 2*self.a*self.d, 1 - 2*tf.square(self.b) - 2*tf.square(self.d), 2*self.c*self.d - 2*self.a*self.b],
            [2*self.b*self.d - 2*self.a*self.c, 2*self.a*self.b + 2*self.c*self.d, 1 - 2*tf.square(self.b) - 2*tf.square(self.c)]
        ])
    
    def to_plotly(self):
        return go.Scatter(
            x=np.array(self.r),
            y=np.array(self.i),
            mode='markers',
            marker={
                'color': ['mediumvioletred']
            }
        )




class Rotate3D:
    def __init__(self, v: UnitVector, angle):
        self.v = v
        self.angle = tf.constant(angle)
    
    def to_rotation_matrix(self):
        cartisan_uv = self.v.to_cartesian()
        a = tf.cos(0.5*self.angle)
        b = tf.sin(0.5*self.angle)*cartisan_uv.x
        c = tf.sin(0.5*self.angle)*cartisan_uv.y
        d = tf.sin(0.5*self.angle)*cartisan_uv.z
        return Quaternion(a,b,c,d).to_rotation_matrix()
        
#     def from_real(self, real: Real):
#         return Rotate2D(tf.atan2(real.i, real.r))
    
#     def to_real(self):
#         return Real(tf.cos(self.theta), tf.sin(self.theta))
    
    def __mul__(self, that):
        pass