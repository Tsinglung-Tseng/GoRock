import argparse
from hotpot.geometry.primiary import Cartesian3
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser(
    description="Count optical photon, aggragate optical simulation data to sample."
)
parser.add_argument("path", type=str, help="Path of imput hits csv file.")

args = parser.parse_args()


def hist_single(hits, eventID, photonID):
    return hits[hits["eventID"] == eventID][
        hits[hits["eventID"] == eventID]["photonID"] == photonID
    ]


def draw_single(single):
    return [
        Cartesian3(a_single["posX"], a_single["posY"], a_single["posZ"]).to_plotly(),
        Cartesian3(
            a_single["sourcePosX"], a_single["sourcePosY"], a_single["sourcePosZ"]
        ).to_plotly(),
    ]


def get_coincidente(hits):
    return (
        hits.groupby("eventID")
        .filter(lambda event: len(event["parentID"].unique()) == 3)["eventID"]
        .unique()
    )


def get_single(hits):
    return (
        hits.groupby("eventID")
        .filter(lambda event: len(event["parentID"].unique()) == 2)["eventID"]
        .unique()
    )


def most_photo_crystal(a_single):
    return a_single.groupby("crystalID").count()["PDGEncoding"].idxmax()


def get_conincidence_hist(hits, eventID):
    single_1 = hits[hits["eventID"] == eventID][
        hits[hits["eventID"] == eventID]["photonID"] == 1
    ]
    single_2 = hits[hits["eventID"] == eventID][
        hits[hits["eventID"] == eventID]["photonID"] == 2
    ]
    nx, ny = (16, 16)
    x = np.linspace(-25, 25, nx)
    y = np.linspace(-25, 25, ny)
    xv, yv = np.meshgrid(x, y)

    return [
        np.histogram2d(single_1.localPosX, single_1.localPosY)[0].tolist(),
        np.histogram2d(single_2.localPosX, single_2.localPosY)[0].tolist(),
    ]


def assemble_samples(hits):
    def _assemble_single_sample(eventID):
        sample = dict(
            hits[hits["eventID"] == eventID][
                ["sourcePosX", "sourcePosY", "sourcePosZ"]
            ].iloc[0]
        )
        sample["counts"] = get_conincidence_hist(hits, eventID)
        sample["crystalID"] = [
            most_photo_crystal(
                hits[hits["eventID"] == eventID][
                    hits[hits["eventID"] == eventID]["photonID"] == 1
                ]
            ),
            most_photo_crystal(
                hits[hits["eventID"] == eventID][
                    hits[hits["eventID"] == eventID]["photonID"] == 2
                ]
            ),
        ]
        sample["eventID"] = eventID
        return sample

    return list(_assemble_single_sample(eventID) for eventID in get_coincidente(hits))


hits_all = pd.read_csv(args.path)
samples = assemble_samples(hits_all)
pd.DataFrame(samples).to_csv("sample.csv", index=False)
