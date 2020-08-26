import plotly.graph_objects as go
import numpy as np
import ipyvolume as ipv


# def point_close_enough(c: Cartisian3):
#     return


# def simple_pair_similar_enough(x, y):
#     return list(map(lambda x: x < 0.00001, np.array([i - j for i, j in zip(x, y)])))


# def shift_loop_array(x):
#     pass


def split_raw_df_into_even_odd_pairs(raw_df):
    return (raw_df[0::2], raw_df[1::2])


class Cartisian3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"""Cartisian3 <size: ({len(self.x)}, {len(self.y)}, {len(self.z)}), x: {self.x}, y: {self.y}, z: {self.z}>"""

    @classmethod
    def from_pattern(self, raw, pattern):
        return Cartisian3(
            x=raw[pattern + "x"], y=raw[pattern + "y"], z=raw[pattern + "z"]
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


def get_source(samples):
    return Cartisian3(
        np.array(samples["source_x"]),
        np.array(samples["source_y"]),
        np.array(samples["source_z"]),
    )


class Pair:
    def __init__(self, fst, snd):
        self.fst = fst
        self.snd = snd


class PairCartisian3(Pair, Cartisian3):
    def __init__(self, raw, pattern):
        source = Cartisian3.from_pattern(raw, pattern)
        Cartisian3.__init__(self, source.x, source.y, source.z)

        fst, snd = split_raw_df_into_even_odd_pairs(raw)
        Pair.__init__(
            self,
            Cartisian3.from_pattern(fst, pattern),
            Cartisian3.from_pattern(snd, pattern),
        )


class Segment(Pair):
    def __init__(self, car: Cartisian3, cdr: Cartisian3):
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
