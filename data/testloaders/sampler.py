import os
import torch
import math
import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler

# sampler used for meta-testing
class random_sampler(Sampler):

    def __init__(self, data_source, way, shot, query_shot=16, trial=1000):

        class2id = {}

        for i, (image_path, class_id) in enumerate(data_source.imgs):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)

        self.class2id = class2id
        self.way = way
        self.shot = shot
        self.trial = trial
        self.query_shot = query_shot

    def __iter__(self):

        way = self.way
        shot = self.shot
        trial = self.trial
        query_shot = self.query_shot

        class2id = deepcopy(self.class2id)
        list_class_id = list(class2id.keys())

        for i in range(trial):

            id_list = []

            np.random.shuffle(list_class_id)
            picked_class = list_class_id[:way]

            for cat in picked_class:
                np.random.shuffle(class2id[cat])

            for cat in picked_class:
                id_list.extend(class2id[cat][:shot])
            for cat in picked_class:
                id_list.extend(class2id[cat][shot:(shot + query_shot)])

            yield id_list