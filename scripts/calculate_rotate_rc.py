import pandas as pd
import numpy as np

from hotpot.geometry.system import FuncDataFrame
from hotpot.geometry.primiary import Cartesian3

from hotpot.database import Database
import os
os.environ["DB_CONNECTION"] ="postgresql://postgres@192.168.1.96:5432/monolithic_crystal"
os.environ["PICLUSTER_DB"] ="postgresql://picluster@192.168.1.96:5432/picluster"

example_hits = pd.read_csv("/home/zengqinglong/optical_simu/5/jiqun_10mm4mm_yuanzhu_9pos/Optical_Syste/simu_80_yuanbing_400sub/sub.0/hits.csv")

crystalID=0
num_points=1000


def solve_rotate_and_offset(source, target):
    """
    `source` : N x 3
    `target` : N x 3
    return : `(R, C)`
    `R` : 3 x 3 matrix
    `C` : 3 vector
    such that `(R @ source.T).T + C = target`
    or equivalently `source @ R.T + C = target`
    """
    a = np.hstack([source, np.ones((source.shape[0], 1))])
    x, y, z = target[:, 0], target[:, 1], target[:, 2]
    c_x = np.linalg.lstsq(a, x, rcond=None)[0]
    c_y = np.linalg.lstsq(a, y, rcond=None)[0]
    c_z = np.linalg.lstsq(a, z, rcond=None)[0]
    R = np.array([c_x[:3], c_y[:3], c_z[:3]])
    C = np.array([c_x[3], c_y[3], c_z[3]])
    return R.tolist(), C.tolist()

result = {
    'crystalID': [],
    'R': [],
    'C': []
}

for crystalID in range(80):
    local_ = FuncDataFrame(example_hits).where(crystalID=crystalID).df[['localPosX','localPosY','localPosZ']].to_numpy()[:num_points]
    global_ = FuncDataFrame(example_hits).where(crystalID=crystalID).df[['posX','posY','posZ']].to_numpy()[:num_points]
    R, C = solve_rotate_and_offset(local_, global_)
    result['crystalID'].append(crystalID)
    result['R'].append(R)
    result['C'].append(C)

pd.DataFrame.from_dict(result).to_sql(con=Database().engine(), name="rotate_rc", if_exists="append", index=False)
