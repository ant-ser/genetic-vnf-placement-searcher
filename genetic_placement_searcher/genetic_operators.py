import random
from abc import abstractmethod
from copy import deepcopy
from typing import Optional, Sequence

import collection_utils
from genetic_algorithm import (
    Chromosome,
    CrossoverOperator,
    InitializationOperator,
    MutationOperator,
    SelectionOperator,
)
from network import NetworkTopology, NetworkNode
from request_accepter import RequestAccepter
from service import VNF
from service_request import ServiceRequest
from vnf_placement import MultiFlavouredVNFChainPlacement


class PlacementInitializationOperator(
    InitializationOperator[
        [
            Sequence[ServiceRequest],
            NetworkTopology,
            Optional[dict[VNF, set[NetworkNode]]],
            float,
        ],
        list[list[int]],
    ]
):
    """
    An abstract base class for initialization operators that generate initial
    placements for genetic algorithms that solve the VNF placement problem.
    """

    @abstractmethod
    def __init__(
        self,
        requests: Sequence[ServiceRequest],
        network: NetworkTopology,
        incompatible_nodes_per_vnf: Optional[dict[VNF, set[NetworkNode]]] = None,
        minimum_ratio_of_main_requests: float = 0,
    ):
        self.requests = requests
        self.network = network
        self.incompatible_nodes_per_vnf = incompatible_nodes_per_vnf
        self.minimum_ratio_of_main_requests = minimum_ratio_of_main_requests


class RandomPlacementInitializationOperator(PlacementInitializationOperator):
    """
    An initialization operator for the VNF placement problem that generates initial
    placements by randomly accepting requests and randomly placing their VNFs on
    suitable nodes in the network.
    """

    def __init__(
        self,
        requests: Sequence[ServiceRequest],
        network: NetworkTopology,
        incompatible_nodes_per_vnf: Optional[dict[VNF, set[NetworkNode]]] = None,
        minimum_ratio_of_main_requests: float = 0,
    ):
        super().__init__(
            requests,
            network,
            incompatible_nodes_per_vnf,
            minimum_ratio_of_main_requests,
        )

    def __call__(self) -> Chromosome[list[list[int]]]:
        empty_placement = MultiFlavouredVNFChainPlacement(
            {request: None for request in self.requests},
            self.network,
            self.incompatible_nodes_per_vnf,
            self.minimum_ratio_of_main_requests,
        )
        request_accepter = RequestAccepter(empty_placement)
        for request in collection_utils.shuffle(self.requests):
            request_accepter.accept(request)
        placement = request_accepter.current_placement
        placement_matrix = placement.placement_matrix
        chromosome = Chromosome(placement_matrix)
        return chromosome


class LinearRankSelectionOperator(SelectionOperator[list[list[int]]]):
    """
    A selection operator for genetic algorithms that sorts all individuals in the
    population by their fitness values and assigns them a rank. The selection
    probability for each individual is then determined based on their rank.
    Specifically, a linear function assigns the highest selection probability to the
    highest-ranked individual and the lowest selection probability to the lowest-ranked
    individual. Linear rank selection can help prevent premature convergence by
    ensuring that even lower-ranked individuals have a chance to be selected and
    contribute to the next generation.

    :param pressure_parameter: This parameter is currently not used, but may be used in
    future implementations.
    """

    def __init__(self, pressure_parameter: float = 1.0) -> None:
        pass

    def __call__(
        self,
        population: Sequence[Chromosome[list[list[int]]]],
        selection_size: Optional[int] = None,
    ) -> list[Chromosome[list[list[int]]]]:
        if selection_size is None:
            selection_size = len(population)
        sorted_population = sorted(population)
        ranks = [index + 1 for index, _ in enumerate(sorted_population)]
        rank_sum = len(ranks) * (len(ranks) + 1) / 2
        selection_probabilities = [rank / rank_sum for rank in ranks]
        return _stochastic_universal_sampling(
            sorted_population, selection_probabilities, selection_size
        )


class ExponentialRankSelectionOperator(SelectionOperator[list[list[int]]]):
    """
    A selection operator for genetic algorithms. Like linear rank selection,
    exponential rank selection first sorts the individuals in the population by their
    fitness values and assigns them a rank. The selection probability for each
    individual is then determined based on their rank, with higher-ranked individuals
    having a higher probability of being selected. However, unlike linear rank
    selection, exponential rank selection uses an exponential function to calculate the
    selection probabilities, which results in a much larger difference in selection
    probabilities between higher-ranked and lower-ranked individuals. This can help
    increase the selective pressure on the population and drive the search towards
    better solutions more quickly.

    :param pressure_parameter: The base of the exponential function used to calculate
        the selection probabilities for each individual. Controls the selective pressure
        applied to the population during the selection process. Must be a value
        in the open interval (0, 1).
        A higher value for this parameter results in a larger difference in selection
        probabilities between higher-ranked and lower-ranked individuals, which
        increases the selective pressure on the population.
    """

    def __init__(self, pressure_parameter: float = 0.9) -> None:
        self.pressure_parameter = pressure_parameter

    def __call__(
        self,
        population: Sequence[Chromosome[list[list[int]]]],
        selection_size: Optional[int] = None,
    ) -> list[Chromosome[list[list[int]]]]:
        if selection_size is None:
            selection_size = len(population)
        if selection_size == 0:
            return []
        sorted_population = sorted(population, reverse=True)
        ranks = [index + 1 for index, _ in enumerate(sorted_population)]
        selection_probabilities = [
            (
                (self.pressure_parameter - 1)
                * (self.pressure_parameter ** (rank - 1))
                / (self.pressure_parameter**selection_size - 1)
            )
            for rank in ranks
        ]
        return _stochastic_universal_sampling(
            sorted_population, selection_probabilities, selection_size
        )


def _stochastic_universal_sampling(
    population: Sequence[Chromosome[list[list[int]]]],
    selection_probabilities: Sequence[float],
    selection_size: int,
) -> list[Chromosome[list[list[int]]]]:
    """
    A sampling technique used in the context of genetic algorithms.
    Stochastic Universal Sampling (SUS) views the cumulative sum of the selection
    probabilities for each individual in the population as a line. The line is split
    into multiple intervals corresponding to the selection probabilities of each
    individual.
    SUS generates a single random value, and places equally spaced pointers on the line
    starting from that value. The number of times an individual appears in the
    resulting list is determined by the number of pointers that land on the interval
    associated with that individual.
    SUS ensures an even spread of selection probabilities and reduces the risk of
    sampling errors. This can help prevent premature convergence and maintain diversity
    in the population.
    """
    if round(sum(selection_probabilities)) != 1:
        raise ValueError("The sum of the elements in probabilities must equal 1")
    probability_sum = 0.0
    pointer = random.uniform(0, 1 / selection_size)
    selected = list[Chromosome[list[list[int]]]]()
    for chromosome, selection_probability in zip(population, selection_probabilities):
        probability_sum += selection_probability
        while pointer < probability_sum:
            selected.append(chromosome)
            pointer += 1 / selection_size
    return selected


class TournamentSelection(SelectionOperator[list[list[int]]]):
    """
    A selection operator for genetic algorithms that randomly chooses a fixed number
    of individuals from the population and selects the individual with the highest
    fitness among them. This process is repeated until the desired number of
    individuals have been selected.
    """

    def __init__(self, tournament_size: int = 2) -> None:
        self.tournament_size = int(tournament_size)

    def __call__(
        self,
        population: Sequence[Chromosome[list[list[int]]]],
        selection_size: Optional[int] = None,
    ) -> list[Chromosome[list[list[int]]]]:
        if selection_size is None:
            selection_size = len(population)
        selection = list[Chromosome[list[list[int]]]]()
        while len(selection) < selection_size:
            candidates = random.choices(population, k=self.tournament_size)
            winner = max(candidates)
            selection.append(winner)
        return selection


class MatrixRowSwapCrossoverOperator(CrossoverOperator[list[list[int]]]):
    """
    A crossover operator for genetic algorithms that solve the VNF placement problem
    that generates new offspring by swapping rows in the placement matrix of two parent
    solutions.
    During the crossover process, a group of mutually exclusive rows from the first
    parent's placement matrix is swapped with the corresponding rows in the second
    parent's placement matrix. If both of the resulting placements are valid, then the
    chromosomes that encode these placements are returned as the new offspring. If
    either of the resulting placements is not valid, then copies of the original parent
    chromosomes are returned instead.
    """

    def __init__(self) -> None:
        pass

    def __call__(
        self, parent1: Chromosome[list[list[int]]], parent2: Chromosome[list[list[int]]]
    ) -> tuple[Chromosome[list[list[int]]], Chromosome[list[list[int]]]]:
        placement1 = parent1.encoded_data
        row_indexes = range(len(placement1.evaluated_requests))
        row_indexes_grouped_by_mutual_exclusivity = (
            self._group_row_indexes_by_mutual_exclusivity(
                row_indexes, placement1.mutually_exclusive_requests_groups
            )
        )
        parent1_genes = deepcopy(parent1.genes)
        parent2_genes = deepcopy(parent2.genes)
        for row_indexes_group in collection_utils.shuffle(
            row_indexes_grouped_by_mutual_exclusivity
        ):
            if any(
                any(gene >= 0 for gene in parent1.genes[row_index])
                or any(gene >= 0 for gene in parent2.genes[row_index])
                for row_index in row_indexes_group
            ):
                (
                    child1_genes,
                    child2_genes,
                ) = collection_utils.swap_elements_inside_interval(
                    parent1_genes,
                    parent2_genes,
                    min(row_indexes_group),
                    max(row_indexes_group) + 1,
                )
                child1 = Chromosome(
                    child1_genes, decoding_function=parent1.decoding_function
                )
                placement1 = child1.encoded_data
                child2 = Chromosome(
                    child2_genes, decoding_function=parent2.decoding_function
                )
                placement2 = child2.encoded_data
                if placement1.is_valid() and placement2.is_valid():
                    break
        else:
            child1 = deepcopy(parent1)
            child2 = deepcopy(parent2)
        return child1, child2

    def _group_row_indexes_by_mutual_exclusivity(
        self,
        row_indexes: Sequence[int],
        mutually_exclusive_requests_groups: list[list[ServiceRequest]],
    ) -> list[list[int]]:
        mutually_exclusive_requests_groups_sizes = [
            len(mutually_exclusive_requests_group)
            for mutually_exclusive_requests_group in mutually_exclusive_requests_groups
        ]
        return collection_utils.chunk(
            row_indexes,
            sizes=mutually_exclusive_requests_groups_sizes,
        )


class RandomAcceptanceMutationOperator(MutationOperator[list[list[int]]]):
    """
    A mutation operator for genetic algorithms that solve the VNF placement problem
    that changes the placement by randomly accepting one or more requests.
    """

    def __init__(self, acceptance_probability: float = 0):
        self.acceptance_probability = acceptance_probability

    def __call__(
        self, chromosome: Chromosome[list[list[int]]]
    ) -> Chromosome[list[list[int]]]:
        placement: MultiFlavouredVNFChainPlacement = chromosome.encoded_data
        request_accepter = RequestAccepter(placement)
        for request in collection_utils.shuffle(placement.evaluated_requests):
            if (
                placement.rejects(request)
                and random.random() < self.acceptance_probability
            ):
                request_accepter.accept(request)
        placement = request_accepter.current_placement
        assert placement.is_valid()
        placement_matrix = placement.placement_matrix
        mutated_chromosome = Chromosome(placement_matrix)
        return mutated_chromosome


class RandomRejectionMutationOperator(MutationOperator[list[list[int]]]):
    """
    A mutation operator for genetic algorithms  that solve the VNF placement problem
    that changes the placement by randomly rejecting one or more requests.
    """

    def __init__(
        self,
        rejection_probability: float = 0,
    ):
        self.rejection_probability = rejection_probability

    def __call__(
        self, chromosome: Chromosome[list[list[int]]]
    ) -> Chromosome[list[list[int]]]:
        placement_matrix = chromosome.genes
        for row_index, row in collection_utils.shuffle(
            list(enumerate(placement_matrix))
        ):
            if (
                all(gene >= 0 for gene in row)
                and random.random() < self.rejection_probability
            ):
                placement_matrix[row_index] = [-1] * len(row)
        mutated_chromosome = Chromosome(placement_matrix)
        mutated_chromosome.decoding_function = chromosome.decoding_function
        return mutated_chromosome


def decoding_function(
    genes: list[list[int]],
    requests: list[ServiceRequest],
    network: NetworkTopology,
    incompatible_nodes_per_vnf: Optional[dict[VNF, set[NetworkNode]]] = None,
    minimum_ratio_of_main_requests: float = 0,
) -> MultiFlavouredVNFChainPlacement:
    """
    Takes as input a list of list of genes representing a placement matrix and uses this
    information to create an instance of the MultiFlavouredVNFChainPlacement class
    that represents the corresponding placement.
    """
    placement_matrix = genes
    placement = MultiFlavouredVNFChainPlacement.from_placement_matrix(
        requests,
        network,
        placement_matrix,
        incompatible_nodes_per_vnf,
        minimum_ratio_of_main_requests,
    )
    return placement
