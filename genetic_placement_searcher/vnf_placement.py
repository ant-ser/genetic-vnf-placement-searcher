import math
from collections import Counter
from functools import cached_property
from typing import Optional, Sequence

import collection_utils
from custom_collections import HashableDict
from network import (
    NetworkTopology,
    NetworkLink,
    NetworkNode,
    NetworkResource,
)
from service import VNF, Service, VirtualLink
from service_request import (
    AlternativeServiceRequest,
    MainServiceRequest,
    ServiceRequest,
)


class VNFChainPlacement:
    def __init__(
        self,
        designated_nodes_per_request: dict[ServiceRequest, Optional[list[NetworkNode]]],
        network_topology: NetworkTopology,
        incompatible_nodes_per_vnf: Optional[dict[VNF, set[NetworkNode]]] = None,
    ):
        self.network_topology = network_topology
        self.designated_nodes_per_request = designated_nodes_per_request
        self.incompatible_nodes_per_vnf = incompatible_nodes_per_vnf

    @classmethod
    def from_placement_matrix(
        cls,
        requests: Sequence[ServiceRequest],
        network_topology: NetworkTopology,
        placement_matrix: list[list[int]],
        incompatible_nodes_per_vnf: Optional[dict[VNF, set[NetworkNode]]] = None,
    ) -> "VNFChainPlacement":
        """
        Creates a placement from a placement matrix. See the property placement_matrix
        of this class for more information on the structure of a placement matrix.
        """
        designated_nodes_per_request = cls._parse_placement_matrix(
            requests, network_topology, placement_matrix
        )
        return VNFChainPlacement(
            designated_nodes_per_request,
            network_topology,
            incompatible_nodes_per_vnf,
        )

    @classmethod
    def _parse_placement_matrix(
        cls,
        requests: Sequence[ServiceRequest],
        network_topology: NetworkTopology,
        placement_matrix: list[list[int]],
    ) -> dict[ServiceRequest, Optional[list[NetworkNode]]]:
        nodes = network_topology.sorted_nodes
        designated_nodes_per_request: dict[
            ServiceRequest, Optional[list[NetworkNode]]
        ] = dict.fromkeys(requests, None)
        for request_index, designated_nodes_indexes in enumerate(placement_matrix):
            if all(node_index >= 0 for node_index in designated_nodes_indexes):
                request = requests[request_index]
                service = request.requested_service
                designated_nodes_per_request[request] = (
                    [service.ingress_node.network_node]
                    + [nodes[node_index] for node_index in designated_nodes_indexes]
                    + [service.egress_node.network_node]
                )
        return designated_nodes_per_request

    @cached_property
    def placement_matrix(self) -> list[list[int]]:
        """
        Returns the placement matrix corresponding to this placement.
        A placement matrix is a list of lists of integers where each row represents a
        request and each column represents a VNF in the VNF chain.
        The values in the matrix represent the index of the node where the
        corresponding VNF should be placed.
        A value of -1 means that the VNF is not placed on any node for that request. As
        a consequence, if a row contains only -1s it means that the corresponding
        request has been rejected.
        """
        placement_matrix = []
        for request, designated_nodes in self.designated_nodes_per_request.items():
            if designated_nodes is None:
                row = [-1] * len(request.requested_service.vnf_chain)
            else:
                row = [
                    self.network_topology.sorted_nodes.index(node)
                    for node in designated_nodes[1:-1]
                ]
            placement_matrix.append(row)
        return placement_matrix

    @cached_property
    def designated_node_per_vnf(self) -> dict[VNF, Optional[NetworkNode]]:
        designated_node_per_vnf = dict[VNF, Optional[NetworkNode]]()
        for request, designated_nodes in self.designated_nodes_per_request.items():
            service = request.requested_service
            vnf_chain = service.vnf_chain
            designated_node_per_vnf.update({vnf: None for vnf in vnf_chain})
            if designated_nodes is not None:
                for vnf, designated_node in zip(vnf_chain, designated_nodes[1:-1]):
                    designated_node_per_vnf[vnf] = designated_node
        return designated_node_per_vnf

    @cached_property
    def vnfs_by_designated_node(self) -> dict[NetworkNode, set[VNF]]:
        vnfs_by_designated_node = {
            node: set[VNF]() for node in self.network_topology.nodes
        }
        for vnf, designated_node in self.designated_node_per_vnf.items():
            if designated_node is not None:
                vnfs_by_designated_node[designated_node].add(vnf)
        return vnfs_by_designated_node

    @cached_property
    def designated_network_link_per_virtual_link(
        self,
    ) -> dict[VirtualLink, Optional[NetworkLink]]:
        designated_network_link_per_virtual_link = dict[
            VirtualLink, Optional[NetworkLink]
        ]()
        for request, designated_nodes in self.designated_nodes_per_request.items():
            service = request.requested_service
            traffic_flow_through_vnfs = service.traffic_flow
            designated_network_link_per_virtual_link.update(
                {virtual_link: None for virtual_link in traffic_flow_through_vnfs.links}
            )
            if designated_nodes is not None:
                designated_network_path = self.network_topology.path_from_nodes(
                    designated_nodes
                )
                for virtual_link, designated_network_link in zip(
                    traffic_flow_through_vnfs.links, designated_network_path.links
                ):
                    designated_network_link_per_virtual_link[
                        virtual_link
                    ] = designated_network_link
        return designated_network_link_per_virtual_link

    @cached_property
    def virtual_links_by_designated_network_link(
        self,
    ) -> dict[NetworkLink, set[VirtualLink]]:
        virtual_links_by_designated_network_link = {
            network_link: set[VirtualLink]()
            for network_link in self.network_topology.links
        }
        for (
            virtual_link,
            network_link,
        ) in self.designated_network_link_per_virtual_link.items():
            if network_link is not None:
                virtual_links_by_designated_network_link[network_link].add(virtual_link)
        return virtual_links_by_designated_network_link

    @cached_property
    def evaluated_requests(self) -> list[ServiceRequest]:
        """
        Returns the evaluated requests in the same order they appear
        in designated_nodes_per_request.
        """
        evaluated_requests = list(self.designated_nodes_per_request.keys())
        return evaluated_requests

    @cached_property
    def accepted_requests(self) -> list[ServiceRequest]:
        accepted_requests = list(filter(self.accepts, self.evaluated_requests))
        return accepted_requests

    @cached_property
    def rejected_requests(self) -> list[ServiceRequest]:
        rejected_requests = list(filter(self.rejects, self.evaluated_requests))
        return rejected_requests

    @cached_property
    def employed_nodes(self) -> set[NetworkNode]:
        designated_nodes = {
            node for node, vnfs in self.vnfs_by_designated_node.items() if vnfs != set()
        }
        return designated_nodes

    @cached_property
    def employed_network_links(self) -> set[NetworkLink]:
        designated_network_links = {
            network_link
            for network_link, virtual_links in (
                self.virtual_links_by_designated_network_link.items()
            )
            if virtual_links != set()
        }
        return designated_network_links

    @cached_property
    def profit(self) -> float:
        profit = self.total_revenue - self.total_costs
        return profit

    @cached_property
    def total_revenue(self) -> float:
        total_revenue = sum(request.revenue for request in self.accepted_requests)
        return total_revenue

    @cached_property
    def total_costs(self) -> float:
        total_costs = self.total_node_costs + self.total_bandwidth_cost
        return total_costs

    @cached_property
    def total_node_costs(self) -> float:
        total_node_costs = sum(
            self.total_cost_per_node[designated_node]
            for designated_node in self.employed_nodes
        )
        return total_node_costs

    @cached_property
    def total_cost_per_node(self) -> dict[NetworkNode, float]:
        resource_unit_costs_per_node = (
            self.network_topology.resource_unit_costs_per_node
        )
        total_cost_per_node = {
            designated_node: sum(
                collection_utils.multiply_numeric_mappings(
                    dict(self.allocated_resources_per_node[designated_node]),
                    resource_unit_costs_per_node[designated_node],
                ).values()
            )
            for designated_node in self.employed_nodes
        }
        return total_cost_per_node

    @cached_property
    def total_bandwidth_cost(self) -> float:
        total_bandwidth_cost = sum(self.total_bandwidth_cost_per_network_link.values())
        return total_bandwidth_cost

    @cached_property
    def total_bandwidth_cost_per_network_link(self) -> dict[NetworkLink, float]:
        bandwidth_unit_costs_per_network_link = (
            self.network_topology.bandwidth_unit_cost_per_link
        )
        total_bandwidth_cost_per_network_link = (
            collection_utils.multiply_numeric_mappings(
                self.allocated_bandwidth_per_network_link,
                bandwidth_unit_costs_per_network_link,
            )
        )
        return total_bandwidth_cost_per_network_link

    @cached_property
    def allocated_resources_per_node(
        self,
    ) -> dict[NetworkNode, Counter[NetworkResource]]:
        allocated_resources_per_node: dict[NetworkNode, Counter[NetworkResource]] = {
            designated_node: sum(
                [vnf.resources_needed_by_resource_type for vnf in vnfs], Counter()
            )
            for designated_node, vnfs in self.vnfs_by_designated_node.items()
        }
        return allocated_resources_per_node

    @cached_property
    def remaining_resources_per_node(
        self,
    ) -> dict[NetworkNode, Counter[NetworkResource]]:
        resource_capacities_per_node = (
            self.network_topology.resource_capacities_per_node
        )
        remaining_resources_per_node = {
            designated_node: collection_utils.subtract_counters(
                resource_capacities_per_node[designated_node],
                self.allocated_resources_per_node[designated_node],
            )
            for designated_node in self.vnfs_by_designated_node.keys()
        }
        return remaining_resources_per_node

    @cached_property
    def cumulative_resource_shortage(self) -> int:
        cumulative_resource_shortage = sum(
            resource_shortage
            for remaining_resources_per_resource_type in (
                self.remaining_resources_per_node.values()
            )
            for remaining_resources in (remaining_resources_per_resource_type.values())
            if (resource_shortage := -remaining_resources) > 0
        )
        return cumulative_resource_shortage

    @cached_property
    def allocated_bandwidth_per_network_link(self) -> dict[NetworkLink, float]:
        allocated_bandwidth_per_network_link = {
            network_link: sum(
                virtual_link.minimum_guaranteed_bandwidth
                for virtual_link in virtual_links
            )
            for network_link, virtual_links in (
                self.virtual_links_by_designated_network_link.items()
            )
        }
        return allocated_bandwidth_per_network_link

    @cached_property
    def remaining_bandwidth_per_network_link(
        self,
    ) -> dict[NetworkLink, float]:
        total_bandwidth_per_network_link = self.network_topology.bandwidth_per_link
        remaining_bandwidth_per_network_link = (
            collection_utils.subtract_numeric_mappings(
                total_bandwidth_per_network_link,
                self.allocated_bandwidth_per_network_link,
            )
        )
        return remaining_bandwidth_per_network_link

    @cached_property
    def cumulative_bandwidth_shortage(self) -> float:
        cumulative_bandwidth_shortage = sum(
            bandwidth_shortage
            for remaining_bandwidth in (
                self.remaining_bandwidth_per_network_link.values()
            )
            if (bandwidth_shortage := -remaining_bandwidth) > 0
        )
        return cumulative_bandwidth_shortage

    @cached_property
    def effective_latencies_by_service(self) -> dict[Service, float]:
        effective_latencies_by_service = dict[Service, float]()
        for request, designated_nodes in self.designated_nodes_per_request.items():
            if designated_nodes is not None:
                service = request.requested_service
                network_path = self.network_topology.path_from_nodes(designated_nodes)
                effective_latencies_by_service[service] = network_path.latency
        return effective_latencies_by_service

    @cached_property
    def cumulative_excess_latency(self) -> float:
        cumulative_excess_latency = sum(
            excess_latency
            for service, service_latency in (
                self.effective_latencies_by_service.items()
            )
            if (excess_latency := service_latency - service.maximum_tolerated_latency)
            > 0
        )
        return cumulative_excess_latency

    @cached_property
    def incompatible_placements(self) -> dict[VNF, NetworkNode]:
        incompatible_placements = {
            vnf: designated_node
            for vnf, designated_node in self.designated_node_per_vnf.items()
            if self.incompatible_nodes_per_vnf is not None
            if vnf in self.incompatible_nodes_per_vnf
            if designated_node in self.incompatible_nodes_per_vnf[vnf]
        }
        return incompatible_placements

    @cached_property
    def incompatible_placements_count(self) -> int:
        count = len(self.incompatible_placements)
        return count

    def accepts(self, request: ServiceRequest) -> bool:
        return self.designated_nodes_per_request[request] is not None

    def rejects(self, request: ServiceRequest) -> bool:
        return not self.accepts(request)

    def is_valid(self) -> bool:
        # The number of digits taken into account when evaluating some of the 
        # constraints that the placement has to satisfy to be considered valid.
        # This is necessary to avoid floating point errors.
        validity_ndigits = 6
        return (
            self.cumulative_resource_shortage == 0
            and round(self.cumulative_excess_latency, validity_ndigits) == 0
            and round(self.cumulative_bandwidth_shortage, validity_ndigits) == 0
            and self.incompatible_placements_count == 0
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VNFChainPlacement):
            return False
        return (
            self.designated_nodes_per_request == other.designated_nodes_per_request
            and self.network_topology == other.network_topology
            and self.incompatible_nodes_per_vnf == other.incompatible_nodes_per_vnf
        )

    def __hash__(self) -> int:
        return hash(
            (
                HashableDict(self.designated_nodes_per_request),
                self.network_topology,
                HashableDict(self.incompatible_nodes_per_vnf or {}),
            )
        )


class MultiFlavouredVNFChainPlacement(VNFChainPlacement):
    def __init__(
        self,
        designated_nodes_per_request: dict[ServiceRequest, Optional[list[NetworkNode]]],
        network_topology: NetworkTopology,
        incompatible_nodes_per_vnf: Optional[dict[VNF, set[NetworkNode]]] = None,
        minimum_ratio_of_main_requests: float = 0,
    ):
        super().__init__(
            designated_nodes_per_request,
            network_topology,
            incompatible_nodes_per_vnf,
        )
        self.minimum_ratio_of_main_requests = minimum_ratio_of_main_requests

    @classmethod
    def from_placement_matrix(
        cls,
        requests: Sequence[ServiceRequest],
        network_topology: NetworkTopology,
        placement_matrix: list[list[int]],
        incompatible_nodes_per_vnf: Optional[dict[VNF, set[NetworkNode]]] = None,
        minimum_ratio_of_main_requests: float = 0,
    ) -> "MultiFlavouredVNFChainPlacement":
        designated_nodes_per_request = cls._parse_placement_matrix(
            requests, network_topology, placement_matrix
        )

        return MultiFlavouredVNFChainPlacement(
            designated_nodes_per_request,
            network_topology,
            incompatible_nodes_per_vnf,
            minimum_ratio_of_main_requests,
        )

    @cached_property
    def evaluated_main_requests(self) -> list[MainServiceRequest]:
        evaluated_main_requests = [
            request
            for request in self.evaluated_requests
            if isinstance(request, MainServiceRequest)
        ]
        return evaluated_main_requests

    @cached_property
    def evaluated_alternative_requests(self) -> list[AlternativeServiceRequest]:
        evaluated_alternative_requests = collection_utils.chain_lists(
            request.alternative_requests for request in self.evaluated_main_requests
        )
        return evaluated_alternative_requests

    @cached_property
    def accepted_main_requests(self) -> list[MainServiceRequest]:
        accepted_main_requests = list(
            filter(self.accepts, self.evaluated_main_requests)
        )
        return accepted_main_requests

    @cached_property
    def rejected_main_requests(self) -> list[MainServiceRequest]:
        rejected_main_requests = list(
            filter(self.rejects, self.evaluated_main_requests)
        )
        return rejected_main_requests

    @cached_property
    def accepted_alternative_requests(self) -> list[AlternativeServiceRequest]:
        accepted_alternative_requests = list(
            filter(self.accepts, self.evaluated_alternative_requests)
        )
        return accepted_alternative_requests

    @cached_property
    def rejected_alternative_requests(self) -> list[AlternativeServiceRequest]:
        rejected_alternative_requests = list(
            filter(self.rejects, self.evaluated_alternative_requests)
        )
        return rejected_alternative_requests

    @cached_property
    def ratio_of_accepted_main_requests(self) -> float:
        accepted_requests_count = len(self.accepted_requests)
        accepted_main_requests_count = len(self.accepted_main_requests)
        if accepted_requests_count == 0:
            ratio = 0.0
        else:
            ratio = accepted_main_requests_count / accepted_requests_count
        return ratio

    @cached_property
    def mutually_exclusive_requests_groups(self) -> list[list[ServiceRequest]]:
        mutually_exclusive_requests_groups = [
            [main_request, *main_request.alternative_requests]
            for main_request in self.evaluated_main_requests
        ]
        return mutually_exclusive_requests_groups

    @cached_property
    def mutual_exclusivity_violations_count(self) -> int:
        mutual_exclusivity_violations = 0
        for (
            mutually_exclusive_requests_group
        ) in self.mutually_exclusive_requests_groups:
            mutually_exclusive_accepted_requests = set(
                mutually_exclusive_requests_group
            ) & set(self.accepted_requests)
            mutual_exclusivity_violations += max(
                0, len(mutually_exclusive_accepted_requests) - 1
            )
        return mutual_exclusivity_violations

    def is_valid(self) -> bool:
        return super().is_valid() and (
            self.mutual_exclusivity_violations_count == 0
            and (
                self.ratio_of_accepted_main_requests
                >= self.minimum_ratio_of_main_requests
            )
        )
