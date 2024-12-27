import numpy as np
import os


def vec_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


class VecDataHandler:
    def __init__(self, vec_size=2048):
        self.vec_size = vec_size
        self.vec_data = np.zeros((1, self.vec_size))

        self.name_list = ['background']  # 此名称对应初始化的全 0 向量

    def add_vector(self, name, vec):
        self.name_list.append(name)
        self.vec_data = np.concatenate([self.vec_data, vec], axis=0)
        assert len(self.vec_data) == len(self.name_list)
