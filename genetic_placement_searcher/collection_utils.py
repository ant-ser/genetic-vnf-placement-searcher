import math
import operator
import random
from collections import Counter
from heapq import nlargest, nsmallest
from itertools import chain, islice
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

T = TypeVar("T")
U = TypeVar("U")


def chain_lists(iterable: Iterable[list[T]]) -> list[T]:
    chained_lists = list(chain.from_iterable(iterable))
    return chained_lists


def join_sets(iterable: Iterable[set[T]]) -> set[T]:
    joined_sets = set().union(*(iterable))
    return joined_sets


def swap_elements_inside_interval(
    input_list_a: list[T],
    input_list_b: list[T],
    start: int,
    end: Optional[int] = None,
) -> Tuple[list[T], list[T]]:
    if end is None:
        end = min(len(input_list_a), len(input_list_b))
    output_list_a = input_list_a[:start] + input_list_b[start:end] + input_list_a[end:]
    output_list_b = input_list_b[:start] + input_list_a[start:end] + input_list_b[end:]
    return output_list_a, output_list_b


def swap_elements_based_on_mask(
    input_list_a: list[T], input_list_b: list[T], mask: list[int]
) -> Tuple[list[T], list[T]]:
    output_list_a = input_list_a.copy()
    output_list_b = input_list_b.copy()
    for index, mask_value in enumerate(mask):
        if mask_value:
            output_list_a[index], output_list_b[index] = (
                input_list_b[index],
                input_list_a[index],
            )
    return output_list_a, output_list_b


def add_counters(counter_a: Counter[T], counter_b: Counter[T]) -> Counter[T]:
    """
    Unlike the '+' operator this function may return Counters that contain negative
    values.
    """
    result = counter_a.copy()
    result = Counter(add_numeric_mappings(dict(counter_a), dict(counter_b)))
    return result


def subtract_counters(counter_a: Counter[T], counter_b: Counter[T]) -> Counter[T]:
    """
    Unlike the '-' operator this function may return Counters that contain negative
    values.
    """
    result = counter_a.copy()
    result.subtract(counter_b)
    return result


def combine_mappings(
    mapping_a: Mapping[T, U],
    mapping_b: Mapping[T, U],
    combination_function: Callable[[U, U], U],
) -> dict[T, U]:
    common_keys = [key for key in mapping_a.keys() if key in mapping_b.keys()]
    result = {
        key: combination_function(mapping_a[key], mapping_b[key]) for key in common_keys
    }
    return result


def add_numeric_mappings(
    mapping_a: Mapping[T, int | float], mapping_b: Mapping[T, int | float]
) -> dict[T, int | float]:
    result = combine_mappings(mapping_a, mapping_b, operator.add)
    return result


def subtract_numeric_mappings(
    mapping_a: Mapping[T, int | float], mapping_b: Mapping[T, int | float]
) -> dict[T, int | float]:
    result = combine_mappings(mapping_a, mapping_b, operator.sub)
    return result


def multiply_numeric_mappings(
    mapping_a: Mapping[T, int | float], mapping_b: Mapping[T, int | float]
) -> dict[T, int | float]:
    result = combine_mappings(mapping_a, mapping_b, operator.mul)
    return result


def divide_numeric_mappings(
    mapping_a: Mapping[T, int | float], mapping_b: Mapping[T, int | float]
) -> dict[T, int | float]:
    def custom_div(dividend: int | float, divisor: int | float) -> int | float:
        if divisor == 0:
            if dividend > 0:
                result = float("inf")
            elif dividend < 0:
                result = float("-inf")
            else:
                result = float("nan")
        else:
            result = dividend / divisor
        return result

    result = combine_mappings(mapping_a, mapping_b, custom_div)
    return result


def chunk(
    sequence: Sequence[T],
    size: Optional[int] = None,
    sizes: Optional[Sequence[int]] = None,
) -> list[list[T]]:
    if size is None and sizes is None:
        raise ValueError("Unspecified chunk size")
    if size is not None and sizes is not None:
        raise ValueError("Arguments size and sizes are mutually exclusive")
    if size is not None:
        sizes = [size] * math.ceil(len(sequence) / size)
    assert sizes is not None
    iterator = iter(sequence)
    chunks = [list(islice(iterator, size)) for size in sizes]
    return chunks


def select_best(
    iterable: Iterable[T], num_best: int, key: Optional[Callable[[T], int]] = None
) -> list[T]:
    return nlargest(num_best, iterable, key=key)


def select_worst(
    iterable: Iterable[T], num_worst: int, key: Optional[Callable[[T], int]] = None
) -> list[T]:
    return nsmallest(num_worst, iterable, key=key)


def shuffle(sequence: Sequence[T]) -> list[T]:
    shuffled_sequence = list(sequence)
    random.shuffle(shuffled_sequence)
    return shuffled_sequence
