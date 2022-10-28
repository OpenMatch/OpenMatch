import random
from typing import List


class DataAugmentationStrategy:

    def augment(self, data: List[int]) -> List[int]:
        raise NotImplementedError

    def __call__(self, data: List[int]) -> List[int]:
        return self.augment(data)


class NullStrategy(DataAugmentationStrategy):

    def augment(self, data: List[int]) -> List[int]:
        return data


class Cropping(DataAugmentationStrategy):

    def __init__(self, ratio_min: float = 0.1, ratio_max: float = 0.5):
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max

    def augment(self, data: List[int]) -> List[int]:
        ratio = random.uniform(self.ratio_min, self.ratio_max)
        length = int(len(data) * ratio)
        start = random.randint(0, len(data) - length)
        end = start + length
        crop = data[start:end]
        return crop


class SequentialStrategies(DataAugmentationStrategy):

    def __init__(self, *strategies: DataAugmentationStrategy):
        self.strategies = strategies

    def add_strategy(self, strategy: DataAugmentationStrategy):
        self.strategies.append(strategy)

    def augment(self, data: List[int]) -> List[int]:
        for strategy in self.strategies:
            data = strategy(data)
        return data
