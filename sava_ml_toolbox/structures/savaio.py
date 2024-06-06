from typing import List


class DectObject:
    def __init__(
        self,
        xywh: List[float],  # 2D bounding box
        xyzwhd: List[float] = [],  # 3D bounding box
        segm: List[float] = [],
        score: float = 0.0,
        label: str = "",
        pc: List[float] = [],
    ) -> None:
        self._xywh: List[float] = xywh
        self._xyzwhd: List[float] = xyzwhd
        self._segm: List[float] = segm
        self._label: str = label
        self._score: float = score
        self._pc: List[float] = pc

    @property
    def xywh(self):
        return self._xywh

    @xywh.setter
    def xywh(self, value):
        self._xywh = value

    @property
    def xyzwhd(self):
        return self._xyzwhd

    @xyzwhd.setter
    def xyzwhd(self, value):
        self._xyzwhd = value

    @property
    def segm(self):
        return self._segm

    @segm.setter
    def segm(self, value):
        self._segm = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

    @property
    def pc(self):
        return self._pc

    @pc.setter
    def pc(self, value):
        self._pc = value


class DetectionListResult:
    def __init__(self, dectobjs: List[DectObject]) -> None:
        self._dectobjs: List[DectObject] = dectobjs

    @property
    def dectobjs(self):
        return self._dectobjs

    @dectobjs.setter
    def dectobjs(self, value):
        self._dectobjs = value
