from functools import reduce
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")


def compose(functions: Sequence[Callable[[T], T]], argument: T) -> T:
    result = reduce(
        lambda arg, fun: fun(arg),
        functions,
        argument,
    )
    return result
