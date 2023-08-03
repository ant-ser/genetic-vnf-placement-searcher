import uuid

from abc import ABC, abstractmethod
from functools import cached_property, total_ordering
from typing import (
    Generic,
    Iterator,
    Sequence,
    Tuple,
    TypeVar,
)


@total_ordering
class Node(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.uuid = uuid.uuid4()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.uuid == other.uuid

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.uuid < other.uuid

    def __hash__(self) -> int:
        return hash(self.uuid)


NodeT = TypeVar("NodeT", bound=Node)


@total_ordering
class Link(Generic[NodeT], ABC):
    @abstractmethod
    def __init__(self, tail: NodeT, head: NodeT) -> None:
        self.tail = tail
        self.head = head

    @property
    def endpoints(self) -> tuple[NodeT, NodeT]:
        return (self.tail, self.head)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Link):
            return False
        return set(self.endpoints) == set(other.endpoints)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Link):
            return False
        return self.endpoints < other.endpoints

    def __hash__(self) -> int:
        return hash(self.endpoints)

    def __contains__(self, obj: object) -> bool:
        return obj in self.endpoints

    def __getitem__(self, index: int) -> NodeT:
        return self.endpoints[index]

    def __iter__(self) -> Iterator[NodeT]:
        return iter(self.endpoints)


LinkT = TypeVar("LinkT", bound=Link)


class Path(Generic[NodeT, LinkT], ABC):
    @abstractmethod
    def __init__(self, links: Sequence[LinkT]):
        self.links = list(links)

    @cached_property
    def origin(self) -> NodeT:
        origin: NodeT = self.links[0][0]
        return origin

    @cached_property
    def destination(self) -> NodeT:
        destination: NodeT = self.links[-1][-1]
        return destination

    @cached_property
    def nodes(self) -> list[NodeT]:
        nodes = [self.origin] + [link[1] for link in self.links]
        return nodes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Path):
            return False
        return self.links == other.links

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Path):
            return False
        return self.links < other.links

    def __hash__(self) -> int:
        return hash(tuple(self.links))

    def __contains__(self, obj: object) -> bool:
        if isinstance(obj, Node):
            return obj in self.nodes
        if isinstance(obj, Link):
            return obj in self.links
        return False


class Topology(Generic[NodeT, LinkT], ABC):
    @abstractmethod
    def __init__(self, nodes: set[NodeT], links: set[LinkT]):
        self.nodes = nodes
        self.links = links

    @cached_property
    def sorted_nodes(self) -> list[NodeT]:
        return sorted(self.nodes)

    @cached_property
    def sorted_links(self) -> list[LinkT]:
        return sorted(self.links)

    @cached_property
    def links_by_endpoints(self) -> dict[Tuple[NodeT, NodeT], LinkT]:
        links_by_endpoints = {link.endpoints: link for link in self.links}
        return links_by_endpoints

    def link(self, tail: NodeT, head: NodeT) -> LinkT:
        link = self.links_by_endpoints[(tail, head)]
        return link

    def incoming_links(self, node: NodeT) -> set[LinkT]:
        return {link for link in self.links if link.head == node}

    def outgoing_links(self, node: NodeT) -> set[LinkT]:
        return {link for link in self.links if link.tail == node}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Topology):
            return False
        return (self.nodes, self.links) == (other.nodes, other.links)

    def __hash__(self) -> int:
        return hash((tuple(self.nodes), tuple(self.links)))
