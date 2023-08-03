import json
from os import path
from functools import partial
from typing import Optional

import genetic_operators
from config import Config
from genetic_algorithm import (
    Chromosome,
    GeneticAlgorithm,
    OperatorSuite,
    Settings,
    TimeLimitTerminationCondition,
)
from input import Input
from fitness_function import VNFPlacementFitnessFunction
from vnf_placement import MultiFlavouredVNFChainPlacement


def search_optimal_placement(
    input_: Input, config: Config
) -> Optional[MultiFlavouredVNFChainPlacement]:
    genetic_algorithm = _setup_genetic_algorithm(input_, config)
    termination_condition = TimeLimitTerminationCondition(config.time_limit)
    initial_population = (
        _parse_initial_population_file(config.initial_population_file_path)
        if config.initial_population_file_path is not None
        else None
    )
    fittest_chromosome = genetic_algorithm.run(
        termination_condition, initial_population
    )
    if not fittest_chromosome.encoded_data.is_valid():
        pass
    placement: Optional[MultiFlavouredVNFChainPlacement] = (
        fittest_chromosome.encoded_data
        if fittest_chromosome.encoded_data.is_valid()
        else None
    )
    return placement


def _setup_genetic_algorithm(
    input_: Input, config: Config
) -> GeneticAlgorithm[list[list[int]]]:
    fitness_function = _setup_fitness_function(config)
    operator_suite = _setup_operator_suite(input_, config)
    genetic_algorithm_settings = _setup_genetic_algorithm_settings(config)
    genetic_algorithm = GeneticAlgorithm(
        fitness_function,
        operator_suite,
        genetic_algorithm_settings,
    )
    return genetic_algorithm


def _setup_fitness_function(config: Config) -> VNFPlacementFitnessFunction:
    fitness_function = VNFPlacementFitnessFunction(config.profit_weight)
    return fitness_function


def _setup_operator_suite(
    input_: Input, config: Config
) -> OperatorSuite[list[list[int]]]:
    operator_suite = OperatorSuite(
        initialization_operator=config.initialization_operator.genetic_operator_type(
            input_.requests,
            input_.network,
            input_.incompatible_nodes_per_vnf,
            input_.minimum_ratio_of_main_requests,
        ),
        selection_operator=config.selection_operator.genetic_operator_type(
            *(config.selection_operator.argument,)
            if config.selection_operator.argument is not None
            else ()
        ),
        crossover_operator=config.crossover_operator.genetic_operator_type(),
        mutation_operators=[
            mutation_operator.genetic_operator_type(
                *(mutation_operator.argument,)
                if mutation_operator.argument is not None
                else ()
            )
            for mutation_operator in config.mutation_operators
        ],
        repair_operators=[
            repair_operator.genetic_operator_type()
            for repair_operator in config.repair_operators
        ],
        decoding_function=partial(
            genetic_operators.decoding_function,
            requests=input_.requests,
            network=input_.network,
            incompatible_nodes_per_vnf=input_.incompatible_nodes_per_vnf,
            minimum_ratio_of_main_requests=input_.minimum_ratio_of_main_requests,
        ),
    )
    return operator_suite


def _setup_genetic_algorithm_settings(config: Config) -> Settings:
    settings = Settings(
        config.population_size,
        config.crossover_probability,
        config.chromosome_mutation_probability,
        config.num_elite,
    )
    return settings


def _parse_initial_population_file(
    initial_population_file_path: str,
) -> list[Chromosome[list[list[int]]]]:
    with open(
        path.normpath(initial_population_file_path), mode="r", encoding="utf-8"
    ) as initial_population_file:
        initial_population = list[Chromosome[list[list[int]]]]()
        for row in initial_population_file:
            genes = json.loads(row)
            chromosome = Chromosome(genes)
            initial_population.append(chromosome)
        return initial_population
