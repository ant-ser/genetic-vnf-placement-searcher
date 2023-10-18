import logging
import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from itertools import zip_longest
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
)

from collection_utils import select_best

main_logger = logging.getLogger("main_logger")
fitness_logger = logging.getLogger("fitness_logger")

InitP = ParamSpec("InitP")
# Type variable that represents the type of chromosome representation used by the
# genetic algorithm
RepT = TypeVar("RepT")


class Chromosome(Generic[RepT]):
    def __init__(
        self,
        genes: RepT,
        fitness_value: float = 0,
        decoding_function: Optional[Callable[[RepT], Any]] = None,
    ):
        self.genes = genes
        self.fitness_value = fitness_value
        self.decoding_function = decoding_function

    @cached_property
    def encoded_data(self) -> Any:
        if self.decoding_function is None:
            raise NotImplementedError()
        return self.decoding_function(self.genes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return False
        return bool(self.genes == other.genes)

    def __hash__(self) -> int:
        return hash(str(self.genes))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Chromosome):
            return False
        return self.fitness_value < other.fitness_value

    def __repr__(self) -> str:
        return "(" + str(self.fitness_value) + ")" + str(self.genes)

    def __deepcopy__(self, memo: Mapping[str, Any]) -> "Chromosome[RepT]":
        copy = Chromosome(
            deepcopy(self.genes), self.fitness_value, self.decoding_function
        )
        if "encoded_data" in self.__dict__:
            copy.__dict__["encoded_data"] = self.encoded_data
        return copy


@dataclass
class Settings:
    def __init__(
        self,
        population_size: int,
        crossover_probability: float,
        chromosome_mutation_probability: float,
        num_elite: int = 0,
    ):
        self.population_size = population_size
        self.num_elites = num_elite
        self.crossover_probability = crossover_probability
        self.chromosome_mutation_probability = chromosome_mutation_probability


class TerminationCondition(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass


class GenerationLimitTerminationCondition(TerminationCondition):
    def __init__(self, total_generations: int) -> None:
        self.current_generation = 0
        self.total_generations = total_generations

    def __bool__(self) -> bool:
        result = self.current_generation >= self.total_generations
        self.current_generation += 1
        return result


class TimeLimitTerminationCondition(TerminationCondition):
    def __init__(self, time_limit: float) -> None:
        self.start_time = time.time()
        self.time_limit = time_limit

    def __bool__(self) -> bool:
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        result = elapsed_time >= self.time_limit
        return result


class FitnessFunction(ABC, Generic[RepT]):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, chromosome: Chromosome[RepT]) -> float:
        pass


class GeneticOperator(ABC, Generic[RepT]):
    @abstractmethod
    def __init__(self) -> None:
        pass


class InitializationOperator(GeneticOperator[RepT], Generic[InitP, RepT]):
    @abstractmethod
    def __init__(self, *init_args: InitP.args, **init_kw_args: InitP.kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self) -> Chromosome[RepT]:
        pass


class SelectionOperator(GeneticOperator[RepT]):
    @abstractmethod
    def __init__(self, parameter: float = 0) -> None:
        pass

    @abstractmethod
    def __call__(
        self, population: Sequence[Chromosome[RepT]], selection_size: Optional[int]
    ) -> list[Chromosome[RepT]]:
        pass


class CrossoverOperator(GeneticOperator[RepT]):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(
        self, parent_a: Chromosome[RepT], parent_b: Chromosome[RepT]
    ) -> tuple[Chromosome[RepT], Chromosome[RepT]]:
        pass


class MutationOperator(GeneticOperator[RepT]):
    @abstractmethod
    def __init__(self, probability: float) -> None:
        pass

    @abstractmethod
    def __call__(self, chromosome: Chromosome[RepT]) -> Chromosome[RepT]:
        pass


class NoOpMutationOperator(MutationOperator[RepT]):
    def __init__(self) -> None:
        pass

    def __call__(self, chromosome: Chromosome[RepT]) -> Chromosome[RepT]:
        return chromosome


class RepairOperator(GeneticOperator[RepT]):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, chromosome: Chromosome[RepT]) -> Chromosome[RepT]:
        pass


class NoOpRepairOperator(RepairOperator[RepT]):
    def __init__(self) -> None:
        pass

    def __call__(self, chromosome: Chromosome[RepT]) -> Chromosome[RepT]:
        return chromosome


class OperatorSuite(Generic[RepT]):
    def __init__(
        self,
        initialization_operator: InitializationOperator[Any, RepT],
        selection_operator: SelectionOperator[RepT],
        crossover_operator: CrossoverOperator[RepT],
        mutation_operators: list[MutationOperator[RepT]],
        repair_operators: list[RepairOperator[RepT]],
        encoding_function: Optional[Callable[[Any], RepT]] = None,
        decoding_function: Optional[Callable[[RepT], Any]] = None,
    ):
        self.initialization_operator = initialization_operator
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.mutation_operators = mutation_operators or [NoOpMutationOperator()]
        self.repair_operators = repair_operators or [NoOpRepairOperator()]
        self.encoding_function = encoding_function
        self.decoding_function = decoding_function


class GeneticAlgorithm(Generic[RepT]):
    def __init__(
        self,
        fitness_function: FitnessFunction[RepT],
        operator_suite: OperatorSuite[RepT],
        settings: Settings,
    ):
        self.fitness_function = fitness_function
        self.operator_suite = operator_suite
        self.settings = settings

    def run(
        self,
        termination_condition: TerminationCondition,
        initial_population: Optional[list[Chromosome[RepT]]] = None,
    ) -> Chromosome[RepT]:
        main_logger.info('Genetic Algorithm execution has begun.')
        current_generation = 0
        population = self._initialize_population(initial_population)
        while not termination_condition:
            main_logger.info(f'Now processing generation No. {current_generation}.')
            population = self._repair_population(population)
            self._evaluate_population(population)
            if (
                main_logger.isEnabledFor(logging.INFO)
                or fitness_logger.isEnabledFor(logging.INFO)
            ):
                fittest_chromosome = max(population)
                main_logger.info(
                    'The fittest chromosome in the current population '
                    f'has a value of {fittest_chromosome.fitness_value}.'
                )
                fitness_logger.info(
                    f"{current_generation},{fittest_chromosome.fitness_value}"
                )
            elite = self._select_elite(population)
            parents = self._select_parents(population)
            children = self._perform_crossovers(parents)
            mutated_children = self._produce_mutations(children)
            population = elite + mutated_children
            current_generation += 1
        main_logger.info('Genetic Algorithm execution has ended.')
        main_logger.info(f'A total of {current_generation} generations were processed.')
        population = self._repair_population(population)
        self._evaluate_population(population)
        fittest_chromosome = max(population)
        main_logger.info(
            'The fittest chromosome of this run has a value of '
            f'{fittest_chromosome.fitness_value}.'
        )
        fitness_logger.info(
            f"{current_generation},{fittest_chromosome.fitness_value}"
        )
        return fittest_chromosome

    def _initialize_population(
        self, initial_population: Optional[list[Chromosome[RepT]]] = None
    ) -> list[Chromosome[RepT]]:
        main_logger.info('Generating the initial population.')
        if initial_population is not None and len(initial_population) != 0:
            main_logger.info(
                'The initial population will contain '
                f'{max(self.settings.population_size, len(initial_population))} '
                'individuals given as input.'
            )
        initial_population = (initial_population or list[Chromosome[RepT]]())[
            : self.settings.population_size
        ]
        while len(initial_population) < self.settings.population_size:
            chromosome = self.operator_suite.initialization_operator()
            initial_population.append(chromosome)
        for chromosome in initial_population:
            chromosome.decoding_function = self.operator_suite.decoding_function
        return initial_population

    def _evaluate_population(self, population: Sequence[Chromosome[RepT]]) -> None:
        main_logger.info('Evaluating the fitness of each individual in the population.')
        for chromosome in population:
            chromosome.fitness_value = self.fitness_function(chromosome)

    def _select_elite(
        self, population: Sequence[Chromosome[RepT]]
    ) -> list[Chromosome[RepT]]:
        if self.settings.num_elites == 0:
            return []
        main_logger.info('Selecting the best individuals as elites.')
        elite = select_best(population, self.settings.num_elites)
        return elite

    def _select_parents(
        self, population: Sequence[Chromosome[RepT]]
    ) -> list[Chromosome[RepT]]:
        main_logger.info('Selecting parents for crossover.')
        settings = self.settings
        selection_size = settings.population_size - settings.num_elites
        parents = self.operator_suite.selection_operator(population, selection_size)
        return parents[: settings.population_size - settings.num_elites]

    def _perform_crossovers(
        self, parents: Sequence[Chromosome[RepT]]
    ) -> list[Chromosome[RepT]]:
        if parents == []:
            return []
        if self.settings.crossover_probability != 0:
            main_logger.info('Generating offspring from parents via crossover.')
        parents = list(parents)
        random.shuffle(parents)
        iterator = iter(parents)
        children = list[Chromosome[RepT]]()
        for parent1, parent2 in zip_longest(iterator, iterator, fillvalue=parents[0]):
            if random.random() < self.settings.crossover_probability:
                child1, child2 = self.operator_suite.crossover_operator(
                    parent1, parent2
                )
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            children += [child1, child2]
        children = children[: len(parents)]
        for child in children:
            child.decoding_function = self.operator_suite.decoding_function
        return children

    def _produce_mutations(
        self, chromosomes: Sequence[Chromosome[RepT]]
    ) -> list[Chromosome[RepT]]:
        if self.settings.chromosome_mutation_probability != 0:
            main_logger.info('Producing mutations on offspring.')
        mutated_chromosomes: list[Chromosome[RepT]] = []
        for chromosome in chromosomes:
            for mutation_operator in self.operator_suite.mutation_operators:
                if random.random() < self.settings.chromosome_mutation_probability:
                    chromosome = mutation_operator(chromosome)
                    chromosome.decoding_function = self.operator_suite.decoding_function
            mutated_chromosomes.append(chromosome)
        return mutated_chromosomes

    def _repair_population(
        self, chromosomes: Sequence[Chromosome[RepT]]
    ) -> list[Chromosome[RepT]]:
        if any(
            not isinstance(repair_operator, NoOpRepairOperator)
            for repair_operator in self.operator_suite.repair_operators
        ):
            main_logger.info('Repairing damaged chromosomes.')
        repaired_chromosomes: list[Chromosome[RepT]] = []
        for chromosome in chromosomes:
            for repair_operator in self.operator_suite.repair_operators:
                chromosome = repair_operator(chromosome)
                chromosome.decoding_function = self.operator_suite.decoding_function
            repaired_chromosomes.append(chromosome)
        return repaired_chromosomes
