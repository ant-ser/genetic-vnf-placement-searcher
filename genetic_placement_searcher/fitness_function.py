from genetic_algorithm import Chromosome, FitnessFunction
from vnf_placement import MultiFlavouredVNFChainPlacement


class VNFPlacementFitnessFunction(FitnessFunction[list[list[int]]]):
    def __init__(self, profit_weight: float):
        self.profit_weight = profit_weight

    def __call__(self, chromosome: Chromosome[list[list[int]]]) -> float:
        placement = chromosome.encoded_data
        fitness_value = self._objective_function(placement)
        return fitness_value

    def _objective_function(self, placement: MultiFlavouredVNFChainPlacement) -> float:
        return self.profit_weight * placement.profit
