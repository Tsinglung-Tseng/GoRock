from .mac import MAC
from ..geometry.primiary import Cartesian3, Box, Trapezoid 
import itertools
import plotly.graph_objects as go
import numpy as np


class ImageSystem(MAC):
    """
    ims = ImageSystem.from_file('/home/zengqinglong/optical_simu/5/jiqun_test_ja_9_2mm/macro/Geometry.mac')
    go.Figure([
        *ims.to_plotly()
    ])
    """

    def __init__(self, mac):
        super().__init__(mac.dump(), "Geometry.mac")

    @staticmethod
    def from_file(path):
        return ImageSystem(MAC.from_file(path))

    @staticmethod
    def from_database(config_id):
        return ImageSystem(MAC.from_database(config_id))

    @property
    def crystal_size(self):
        return [
            float(i.split()[0])
            for i in self.to_json()["gate"]["crystal"]["geometry"].values()
        ]

    @property
    def translation(self):
        return [
            float(i)
            for i in self.to_json()["gate"]["crystal"]["placement"][
                "setTranslation"
            ].split()[:-1]
        ]

    @property
    def linearRepeatNumber(self):
        return int(self.to_json()["gate"]["crystal"]["linear"]["setRepeatNumber"])

    @property
    def linearRepeatVector(self):
        return [
            float(i)
            for i in self.to_json()["gate"]["crystal"]["linear"][
                "setRepeatVector"
            ].split()[:-1]
        ]

    @property
    def ringRepeatNumber(self):
        return int(self.to_json()["gate"]["crystal"]["ring"]["setRepeatNumber"])

    @property
    def rotate_rpy_mask(self):
        return [
            float(i)
            for i in self.to_json()["gate"]["crystal"]["ring"]["setPoint1"].split()[:-1]
        ]

    @property
    def ring_rv(self):
        return (
            Cartesian3(
                Cartesian3.from_tuple3s(
                    [
                        np.repeat(2 * np.pi / self.ringRepeatNumber * i, 3).tolist()
                        for i in range(self.ringRepeatNumber)
                    ]
                ).x
                * Cartesian3.from_tuple(self.rotate_rpy_mask).x,
                Cartesian3.from_tuple3s(
                    [
                        np.repeat(2 * np.pi / self.ringRepeatNumber * i, 3).tolist()
                        for i in range(self.ringRepeatNumber)
                    ]
                ).y
                * Cartesian3.from_tuple(self.rotate_rpy_mask).y,
                Cartesian3.from_tuple3s(
                    [
                        np.repeat(2 * np.pi / self.ringRepeatNumber * i, 3).tolist()
                        for i in range(self.ringRepeatNumber)
                    ]
                ).z
                * Cartesian3.from_tuple(self.rotate_rpy_mask).z,
            )
            .to_tensor()
            .numpy()
            .T.tolist()
        )

    @property
    def linear_mv(self):
        step = np.array(self.linearRepeatVector)
        result = []
        
        origin_correction = step * (self.linearRepeatNumber-1) / 2

        for i in np.arange(self.linearRepeatNumber):
            result.append(i * step)
        return (np.array(result)-origin_correction).tolist()

    # @property
    # def linear_mv(self):
        # step = np.array(self.linearRepeatVector)
        # base = step / 2
        # neg_base = base * -1
        # result = []

        # for i in range(1, int(self.linearRepeatNumber / 2 + 1)):
            # result.append((base + (i - 1) * step).tolist())
            # result.append((neg_base - (i - 1) * step).tolist())
        # return result

    @property
    def image_system_mr_paras(self):
        return [
            (self.translation, *i)
            for i in list(itertools.product(self.linear_mv, self.ring_rv))
        ]

    def to_plotly(self):
        boxies = [
            Box.from_size(*self.crystal_size)
            .move(para[0])
            .move(para[1])
            .rotate_ypr(para[2])
            .to_plotly()
            for para in self.image_system_mr_paras
        ]
        return [item for sublist in boxies for item in sublist]


class AlbiraImageSystem(ImageSystem):
    def __init__(self, mac):
        super().__init__(mac)

    @staticmethod
    def from_file(path):
        return AlbiraImageSystem(MAC.from_file(path))

    def to_plotly(self):
        boxies = [
            Trapezoid(
                Trapezoid.from_size(
                    self.crystal_size[0],
                    self.crystal_size[2],
                    self.crystal_size[4]
                ).vertices
                .move(para[0])
                .move(para[1])
                .rotate_ypr(para[2])
            )
            .to_plotly()
            for para in self.image_system_mr_paras
        ]
        return sum(boxies, [])
