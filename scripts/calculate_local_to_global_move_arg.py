import pandas as pd
import numpy as np
import tensorflow as tf

from hotpot.geometry.system import FuncDataFrame
from hotpot.geometry.primiary import Cartesian3

import functools
from hotpot.database import Database
from pifs.spack_util import FuncList
from itertools import combinations, chain
import os

os.environ[
    "DB_CONNECTION"
] = "postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
os.environ["PICLUSTER_DB"] = "postgresql://picluster@192.168.1.96:5432/picluster"

example_hits = pd.read_csv(
    "/home/zengqinglong/optical_simu/5/jiqun_10mm4mm_yuanzhu_9pos/Optical_Syste/simu_80_yuanbing_400sub/sub.0/hits.csv"
)

example_hits = example_hits[:136413]

all_possible_moves = (
    FuncList(combinations([30, 90, -30, -90], 1))
    .map(lambda x: [0, *x, 217.5])
    .map(lambda x: [x, list(np.pi / 10 * np.arange(20))])
    .map(lambda x: [[*x[0], 0, i, 0] for i in x[1]])
    .to_list()
)

all_possible_moves = np.reshape(np.array(all_possible_moves), (80, 6)).tolist()


def rotation_matrix(ypr):
    def rotation_matrix_x(angle):
        return np.matrix(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )

    def rotation_matrix_y(angle):
        return np.matrix(
            [
                [np.cos(angle), 0, np.sin(angle)],
                [0.0, 1.0, 0.0],
                [-np.sin(angle), 0, np.cos(angle)],
            ]
        )

    def rotation_matrix_z(angle):
        return np.matrix(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

    return (
        rotation_matrix_x(ypr[0])
        * rotation_matrix_y(ypr[1])
        * rotation_matrix_z(ypr[2])
    )


def close_enough(cart_1, cart_2):
    coordinate_close = (
        (cart_1 - cart_2)
        .fmap(tf.abs)
        .fmap(lambda i: i < 0.001)
        .fmap(lambda i: tf.reshape(i, [-1]))
        .fmap(lambda i: [functools.reduce(lambda x, y: x & y, i.numpy())])
    )
    if (
        coordinate_close.x == True
        and coordinate_close.y == True
        and coordinate_close.z == True
    ):
        return True
    else:
        return False


def local_pos_2_global_pos(local_pos, move_args):
    move = move_args[:3]
    rotate_mat = rotation_matrix(move_args[3:])

    return local_pos.move(move).left_matmul(rotate_mat)


tmp = []
for crystalID in range(80):
    raw = FuncDataFrame(example_hits).where(crystalID=crystalID).df

    global_pos = Cartesian3.pos_from_hits(raw)

    local_pos = Cartesian3.local_pos_from_hits(raw)

    result = [
        (close_enough(local_pos_2_global_pos(local_pos, ma), global_pos), ma)
        for ma in all_possible_moves
    ]
    tmp.append(
        [crystalID, *FuncList(result).filter(lambda i: i[0] is True).to_list()[0][1]]
    )


pd.DataFrame(tmp).to_csv("local_to_global.csv", index=False)
