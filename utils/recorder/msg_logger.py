# -*- coding: utf-8 -*-



class TxtLogger(object):
    __slots__ = ["_path"]

    def __init__(self, path):
        self._path = path

    def record(self, msg, show=True):
        with open(self._path, "a") as f:
            f.write(str(msg) + "\n")

        if show:
            print(msg)
