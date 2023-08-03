from collections import Counter
from functools import cached_property, total_ordering
from itertools import pairwise
from typing import Sequence

from topology import Link, Topology, Node, Path


@total_ordering
class NetworkResource:
    def __init__(self, resource_type: str) -> None:
        self.resource_type = resource_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NetworkResource):
            return False
        return self.resource_type == other.resource_type

    def __hash__(self) -> int:
        return hash(self.resource_type)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, NetworkResource):
            return False
        return (len(self.resource_type), self.resource_type) < (
            len(other.resource_type),
            other.resource_type,
        )


@total_ordering
class NetworkNode(Node):
    def __init__(
        self,
        label: str,
        capacity_per_resource_type: Counter[NetworkResource],
        unit_cost_per_resource_type: dict[NetworkResource, float],
    ):
        super().__init__()
        self.label = label
        self.capacity_per_resource_type = capacity_per_resource_type
        self.unit_cost_per_resource_type = unit_cost_per_resource_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NetworkNode):
            return False
        return self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, NetworkNode):
            return False
        return (len(self.label), self.label) < (len(other.label), other.label)


class NetworkLink(Link[NetworkNode]):
    def __init__(
        self,
        tail: NetworkNode,
        head: NetworkNode,
        latency: float = 0,
        bandwidth: float = 0,
        bandwidth_unit_cost: float = 0,
    ):
        super().__init__(tail, head)
        self.latency = latency
        self.bandwidth = bandwidth
        self.bandwidth_unit_cost = bandwidth_unit_cost


class NetworkPath(Path[NetworkNode, NetworkLink]):
    def __init__(self, links: Sequence[NetworkLink]):
        super().__init__(links)

    @cached_property
    def latency(self) -> float:
        path_latency = sum(link.latency for link in self.links)
        return path_latency


class NetworkTopology(Topology[NetworkNode, NetworkLink]):
    def __init__(self, nodes: set[NetworkNode], links: set[NetworkLink]):
        super().__init__(nodes, links)

    @cached_property
    def resource_capacities_per_node(
        self,
    ) -> dict[NetworkNode, Counter[NetworkResource]]:
        resource_capacities_per_node = {
            node: node.capacity_per_resource_type for node in self.nodes
        }
        return resource_capacities_per_node

    @cached_property
    def resource_unit_costs_per_node(
        self,
    ) -> dict[NetworkNode, dict[NetworkResource, float]]:
        resource_unit_costs_per_node = {
            node: node.unit_cost_per_resource_type for node in self.nodes
        }
        return resource_unit_costs_per_node

    @cached_property
    def latency_per_link(self) -> dict[NetworkLink, float]:
        latency_per_link = {link: link.latency for link in self.links}
        return latency_per_link

    @cached_property
    def bandwidth_per_link(self) -> dict[NetworkLink, float]:
        bandwidth_per_link = {link: link.bandwidth for link in self.links}
        return bandwidth_per_link

    @cached_property
    def bandwidth_unit_cost_per_link(self) -> dict[NetworkLink, float]:
        bandwidth_unit_cost_per_link = {
            link: link.bandwidth_unit_cost for link in self.links
        }
        return bandwidth_unit_cost_per_link

    def path_from_nodes(self, nodes: list[NetworkNode]) -> NetworkPath:
        links = [self.link(*endpoints) for endpoints in pairwise(nodes)]
        return self.path_from_links(links)

    def path_from_links(self, links: list[NetworkLink]) -> NetworkPath:
        return NetworkPath(links)
