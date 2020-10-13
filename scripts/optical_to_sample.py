import argparse
from hotpot.geometry.primiary import Cartesian3
from hotpot.geometry.system import Crystal, SipmArray, Hit
import pandas as pd
import numpy as np
import os

os.environ[
    "DB_CONNECTION"
] = "postgresql://zengqinglong@192.168.1.96:5432/monolithic_crystal"
os.environ["PICLUSTER_DB"] = "postgresql://picluster@192.168.1.96:5432/picluster"


parser = argparse.ArgumentParser(
    description="Count optical photon, aggragate optical simulation data to sample."
)
parser.add_argument("path", type=str, help="Path of imput hits csv file.")

args = parser.parse_args()

hits = pd.read_csv(args.path)
h = Hit(hits)
c_samples = h.coincidence.coincidence_sample()
c_samples.to_csv("sample.csv", index=False)
