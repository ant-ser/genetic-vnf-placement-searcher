import os
import re
from configparser import ConfigParser, ParsingError
from typing import (
    Generic,
    Mapping,
    Optional,
    Type,
    TypeVar,
)


from genetic_algorithm import (
    CrossoverOperator,
    MutationOperator,
    RepairOperator,
    SelectionOperator,
)
from genetic_operators import (
    ExponentialRankSelectionOperator,
    LinearRankSelectionOperator,
    MatrixRowSwapCrossoverOperator,
    PlacementInitializationOperator,
    RandomAcceptanceMutationOperator,
    RandomPlacementInitializationOperator,
    RandomRejectionMutationOperator,
    TournamentSelection,
)


GeneticOperatorT = TypeVar("GeneticOperatorT")


class GeneticOperatorConfiguration(Generic[GeneticOperatorT]):
    def __init__(
        self,
        genetic_operator_type: Type[GeneticOperatorT],
        argument: Optional[float] = None,
    ):
        self.genetic_operator_type = genetic_operator_type
        self.argument = argument


class CustomConfigParser(ConfigParser):
    SUPPORTED_INITIALIZATION_OPERATORS: dict[
        str, Type[PlacementInitializationOperator]
    ] = {
        "RandomInitialization": RandomPlacementInitializationOperator,
    }
    SUPPORTED_SELECTION_OPERATORS: dict[
        str, Type[SelectionOperator[list[list[int]]]]
    ] = {
        "LinearRankSelection": LinearRankSelectionOperator,
        "ExponentialRankSelection": ExponentialRankSelectionOperator,
        "TournamentSelection": TournamentSelection,
    }
    SUPPORTED_CROSSOVER_OPERATORS: dict[
        str, Type[CrossoverOperator[list[list[int]]]]
    ] = {
        "RowSwapCrossover": MatrixRowSwapCrossoverOperator,
    }
    SUPPORTED_MUTATION_OPERATORS: dict[str, Type[MutationOperator[list[list[int]]]]] = {
        "RandomAcceptanceMutation": RandomAcceptanceMutationOperator,
        "RandomRejectionMutation": RandomRejectionMutationOperator,
    }
    SUPPORTED_REPAIR_OPERATORS: dict[str, Type[RepairOperator[list[list[int]]]]] = {}

    def __init__(self, config_file_path: str, encoding: Optional[str] = None):
        if not os.path.isfile(config_file_path):
            raise FileNotFoundError("Configuration file not found")
        converters = {
            "initializationoperator": lambda string: self.parse_string_as_operator(
                string,
                operator_namespace=self.SUPPORTED_INITIALIZATION_OPERATORS,
            ),
            "selectionoperator": lambda string: self.parse_string_as_operator(
                string,
                operator_namespace=self.SUPPORTED_SELECTION_OPERATORS,
            ),
            "crossoveroperator": lambda string: self.parse_string_as_operator(
                string,
                operator_namespace=self.SUPPORTED_CROSSOVER_OPERATORS,
            ),
            "mutationoperators": lambda string: self.parse_string_as_list_of_operators(
                string,
                operator_namespace=self.SUPPORTED_MUTATION_OPERATORS,
            ),
            "repairoperators": lambda string: self.parse_string_as_list_of_operators(
                string,
                operator_namespace=self.SUPPORTED_REPAIR_OPERATORS,
            ),
        }
        super().__init__(converters=converters)
        super().read(config_file_path, encoding)

    @classmethod
    def parse_string_as_list_of_operators(
        cls,
        string: str,
        operator_namespace: Mapping[str, Type[GeneticOperatorT]],
    ) -> list[GeneticOperatorConfiguration[GeneticOperatorT]]:
        operator_strings = [elem.strip() for elem in string.split(",")]
        operators = [
            operator_type
            for elem in operator_strings
            if (operator_type := cls.parse_string_as_operator(elem, operator_namespace))
            is not None
        ]
        return operators

    @classmethod
    def parse_string_as_operator(
        cls,
        string: str,
        operator_namespace: Mapping[str, Type[GeneticOperatorT]],
    ) -> Optional[GeneticOperatorConfiguration[GeneticOperatorT]]:
        operator_name_pattern = "[_a-zA-Z][_\\w]*"
        operator_argument_pattern = "\\d+(.\\d+)?"
        complete_operator_pattern = (
            f"(?:({operator_name_pattern})(?:\\(({operator_argument_pattern})\\))?)?"
        )
        match = re.fullmatch(complete_operator_pattern, string)
        if match is None:
            raise ParsingError("Operator is invalid")
        if match.string == "":
            return None
        operator_name = match.groups()[0]
        operator_argument = (
            float(match.groups()[1]) if match.groups()[1] is not None else None
        )
        operator_type = operator_namespace[operator_name]
        return GeneticOperatorConfiguration(operator_type, operator_argument)


class Config:
    def __init__(
        self,
        profit_weight: float,
        initialization_operator: GeneticOperatorConfiguration[
            PlacementInitializationOperator
        ],
        selection_operator: GeneticOperatorConfiguration[
            SelectionOperator[list[list[int]]]
        ],
        crossover_operator: GeneticOperatorConfiguration[
            CrossoverOperator[list[list[int]]]
        ],
        mutation_operators: list[
            GeneticOperatorConfiguration[MutationOperator[list[list[int]]]]
        ],
        repair_operators: list[
            GeneticOperatorConfiguration[RepairOperator[list[list[int]]]]
        ],
        population_size: int,
        time_limit: float,
        crossover_probability: float,
        chromosome_mutation_probability: float,
        num_elite: int,
        initial_population_file_path: Optional[str] = None,
    ):
        self.profit_weight = profit_weight
        self.initialization_operator = initialization_operator
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.mutation_operators = mutation_operators
        self.repair_operators = repair_operators
        self.population_size = population_size
        self.time_limit = time_limit
        self.crossover_probability = crossover_probability
        self.chromosome_mutation_probability = chromosome_mutation_probability
        self.num_elite = num_elite
        self.initial_population_file_path = initial_population_file_path

    @classmethod
    def from_file(cls, config_file_path: str) -> "Config":
        config_parser = CustomConfigParser(config_file_path, encoding="ascii")
        fitness_function_section = "Fitness_Function_Settings"
        operators_section = "Operator_Settings"
        general_section = "General_Settings"
        return Config(
            config_parser.getfloat(fitness_function_section, "profit_weight"),
            config_parser.getinitializationoperator(
                operators_section, "initialization_operator"
            ),
            config_parser.getselectionoperator(operators_section, "selection_operator"),
            config_parser.getcrossoveroperator(operators_section, "crossover_operator"),
            config_parser.getmutationoperators(operators_section, "mutation_operators"),
            config_parser.getrepairoperators(operators_section, "repair_operators"),
            config_parser.getint(general_section, "population_size"),
            config_parser.getfloat(general_section, "time_limit"),
            config_parser.getfloat(general_section, "crossover_probability"),
            config_parser.getfloat(general_section, "chromosome_mutation_probability"),
            config_parser.getint(general_section, "num_elite"),
            config_parser.get(
                general_section, "initial_population_file_path", fallback=None
            ),
        )
