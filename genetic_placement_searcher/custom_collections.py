from typing import Any, Optional, Tuple, TypeVar

KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")
DefaultValueT = TypeVar("DefaultValueT")


class ImmutableDict(dict[KeyT, ValueT]):
    def __delitem__(self, key: KeyT) -> None:
        raise TypeError(
            f"{self.__class__.__name__} object does not support item deletion"
        )

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        raise TypeError(
            f"{self.__class__.__name__} object does not support item assignment"
        )

    def clear(self) -> None:
        raise TypeError(f"{self.__class__.__name__} object does not support clearing")

    def pop(self, key: Any = None, default: Any = None) -> ValueT | DefaultValueT:
        raise TypeError(f"{self.__class__.__name__} object does not support popping")

    def popitem(self) -> Tuple[KeyT, ValueT]:
        raise TypeError(
            f"{self.__class__.__name__} object does not support popping items"
        )

    def setdefault(self, key: KeyT, default: Optional[ValueT] = None) -> ValueT:
        raise TypeError(
            f"{self.__class__.__name__} object does not support setting default values"
        )

    def update(self, __other: Any = None, **_kwargs: Any) -> None:
        raise TypeError(f"{self.__class__.__name__} object does not support updating")


class HashableDict(ImmutableDict[KeyT, ValueT]):
    def __hash__(self) -> int:
        return hash(frozenset(self.items()))
