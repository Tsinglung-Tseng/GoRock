import argparse
from hotpot.geometry.primiary import Cartesian3
import pandas as pd
import numpy as np
from plotly import graph_objs as go
from hotpot.database import Database
import matplotlib.pyplot as plt
from itertools import combinations, chain
import itertools
import functools
import operator
from pifs.spack_util import FuncList

import os

os.environ[
    "DB_CONNECTION"
] = "postgresql://zengqinglong:zqlthedever@192.168.1.96:5432/monolithic_crystal"

import tensorflow as tf
from hotpot.utils.tf_gpu import USEGPU

USEGPU(0)

raw = Database().read_sql(
    """SELECT
	*
FROM
	hits
WHERE "eventID" = 120 AND "crystalID"=0;"""
)

global_pos = Cartesian3.pos_from_hits(raw)

local_pos = Cartesian3.local_pos_from_hits(raw)


xxx = (
    FuncList(combinations([30, 90, -30, -90], 1))
    .map(lambda x: [0, *x, 217.5])
    .map(lambda x: [x, list(np.pi / 10 * np.arange(20))])
    .map(lambda x: [[*x[0], i] for i in x[1]])
    .to_list()
)

move_args = [*xxx[0], *xxx[1], *xxx[2], *xxx[3]]


def rotation_matrix_y(angle):
    return tf.convert_to_tensor(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )


angles = np.arange(0, 360, 20) * np.pi / 180
all_rotate_matrix = [rotation_matrix_y(a) for a in angles]


def close_enough(cart_1, cart_2):
    coordinate_close = (
        (cart_1 - cart_2)
        .fmap(tf.abs)
        .fmap(lambda i: i < 0.001)
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
    rotate_mat = rotation_matrix_y(move_args[3:][0])

    return local_pos.move(move).rotate_using_rotate_matrix(rotate_mat)


# result = [(close_enough(local_pos_2_global_pos(local_pos, ma), global_pos), ma) for ma in move_args]

# FuncList(result).filter(lambda i: i[0] is True).to_list()

# crystal_transfer_args = pd.DataFrame()

tmp = []
for crystalID in range(80):
    raw = Database().read_sql(
        f"""SELECT
        *
    FROM
        hits
    WHERE "eventID" = 120 AND "crystalID"={crystalID};"""
    )

    global_pos = Cartesian3.pos_from_hits(raw)

    local_pos = Cartesian3.local_pos_from_hits(raw)

    result = [
        (close_enough(local_pos_2_global_pos(local_pos, ma), global_pos), ma)
        for ma in move_args
    ]
    tmp.append(
        [crystalID, *FuncList(result).filter(lambda i: i[0] is True).to_list()[0][1]]
    )


pd.DataFrame(tmp).to_csv("local_to_global.csv", index=False)
