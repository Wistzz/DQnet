# -*- coding: utf-8 -*-

class AvgMeter(object):
    __slots__ = ["value", "avg", "sum", "count"]

    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, num=1):
        self.value = value
        self.sum += value * num
        self.count += num
        self.avg = self.sum / self.count
