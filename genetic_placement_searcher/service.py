from collections import Counter
from functools import cached_property, total_ordering
from itertools import pairwise

from network import NetworkNode, NetworkResource
from topology import Link, Topology, Node, Path


class VirtualNode(Node):
    pass


class VNF(VirtualNode):
    def __init__(self, resources_needed_by_resource_type: Counter[NetworkResource]):
        super().__init__()
        self.resources_needed_by_resource_type = resources_needed_by_resource_type


class ServiceEndpoint(VirtualNode):
    def __init__(self, network_node: NetworkNode):
        super().__init__()
        self.network_node = network_node


@total_ordering
class VirtualLink(Link[VirtualNode]):
    def __init__(
        self,
        tail: VirtualNode,
        head: VirtualNode,
        minimum_guaranteed_bandwidth: float,
    ):
        super().__init__(tail, head)
        self.minimum_guaranteed_bandwidth = minimum_guaranteed_bandwidth


class TrafficFlow(Path[VirtualNode, VirtualLink]):
    def __init__(self, links: list[VirtualLink], latency: float):
        super().__init__(links)
        self.latency = latency


class ServiceTopology(Topology[VirtualNode, VirtualLink]):
    def __init__(
        self,
        ingress_node: ServiceEndpoint,
        egress_node: ServiceEndpoint,
        vnf_chain: list[VNF],
        links: set[VirtualLink],
    ):
        nodes = {ingress_node, *vnf_chain, egress_node}
        super().__init__(nodes, links)
        self.ingress_node = ingress_node
        self.egress_node = egress_node
        self.vnf_chain = vnf_chain

    @cached_property
    def resources_needed_per_vnf(self) -> dict[VNF, Counter[NetworkResource]]:
        resources_needed_per_vnf = {
            vnf: vnf.resources_needed_by_resource_type for vnf in self.vnf_chain
        }
        return resources_needed_per_vnf

    @cached_property
    def bandwidth_per_virtual_link(self) -> dict[VirtualLink, float]:
        bandwidth_per_virtual_link = {
            virtual_link: virtual_link.minimum_guaranteed_bandwidth
            for virtual_link in self.links
        }
        return bandwidth_per_virtual_link


class Service:
    def __init__(
        self,
        ingress_node: ServiceEndpoint,
        egress_node: ServiceEndpoint,
        vnf_chain: list[VNF],
        links: set[VirtualLink],
        maximum_tolerated_latency: float,
    ):
        self.service_topology = ServiceTopology(
            ingress_node, egress_node, vnf_chain, links
        )
        self.maximum_tolerated_latency = maximum_tolerated_latency

    @cached_property
    def ingress_node(self) -> ServiceEndpoint:
        return self.service_topology.ingress_node

    @cached_property
    def egress_node(self) -> ServiceEndpoint:
        return self.service_topology.egress_node

    @cached_property
    def vnf_chain(self) -> list[VNF]:
        return self.service_topology.vnf_chain

    @cached_property
    def resources_needed_per_vnf(self) -> dict[VNF, Counter[NetworkResource]]:
        resources_needed_per_vnf = self.service_topology.resources_needed_per_vnf
        return resources_needed_per_vnf

    @cached_property
    def bandwidth_per_virtual_link(self) -> dict[VirtualLink, float]:
        bandwidth_per_virtual_link = self.service_topology.bandwidth_per_virtual_link
        return bandwidth_per_virtual_link

    @cached_property
    def traffic_flow(self) -> TrafficFlow:
        nodes = [self.ingress_node, *self.vnf_chain, self.egress_node]
        virtual_links = [
            self.service_topology.link(*endpoints) for endpoints in pairwise(nodes)
        ]
        return TrafficFlow(virtual_links, self.maximum_tolerated_latency)
