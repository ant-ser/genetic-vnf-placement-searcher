from copy import copy
from typing import Optional

import collection_utils
from network import NetworkLink, NetworkNode
from service import VNF, VirtualLink
from service_request import (
    MainServiceRequest,
    ServiceRequest,
)
from vnf_placement import MultiFlavouredVNFChainPlacement


class RequestAccepter:
    """
    This class is used to update a placement by accepting requests that are currently
    rejected.
    Before accepting a request, the class checks if it is compatible with other
    requests that are currently accepted. Then, it finds suitable nodes for placing the
    VNFs of the requested service while ensuring that the placement remains valid.
    """

    def __init__(self, initial_placement: MultiFlavouredVNFChainPlacement) -> None:
        self.current_placement = initial_placement
        self._remaining_resources_per_node = copy(
            self.current_placement.remaining_resources_per_node
        )
        self._remaining_bandwidth_per_network_link = copy(
            self.current_placement.remaining_bandwidth_per_network_link
        )

    def accept(self, request: ServiceRequest) -> bool:
        """
        Accepts a request and updates the current placement accordingly.
        A request cannot be accepted if:
            - It is not present in the initial placement.
            - It is already accepted.
            - It is not compatible with currently accepted requests.
            - There are no suitable nodes for placing the VNFs of the requested service.
        If the specified request has been accepted, this method returns True, otherwise,
        it returns False.
        """
        designated_nodes_per_request = (
            self.current_placement.designated_nodes_per_request
        )
        if request not in designated_nodes_per_request:
            raise ValueError(
                "Only requests present in the initial placement can be accepted."
            )
        if designated_nodes_per_request[request] is not None:
            raise ValueError(f"Request {request} is already been accepted.")
        if self._is_request_compatible_with_currently_accepted_requests(request):
            suitable_nodes = self._find_suitable_nodes_for_request(request)
            if suitable_nodes is not None:
                self.current_placement = MultiFlavouredVNFChainPlacement(
                    designated_nodes_per_request | {request: suitable_nodes},
                    self.current_placement.network_topology,
                    self.current_placement.incompatible_nodes_per_vnf,
                    self.current_placement.minimum_ratio_of_main_requests,
                )
                return True
        return False

    def _is_request_compatible_with_currently_accepted_requests(
        self, request: ServiceRequest
    ) -> bool:
        return self._are_mutually_exclusive_requests_rejected(request) and (
            self._would_accepting_request_meet_minimum_main_request_ratio(request)
        )

    def _are_mutually_exclusive_requests_rejected(
        self, request: ServiceRequest
    ) -> bool:
        mutually_exclusive_requests_groups = (
            self.current_placement.mutually_exclusive_requests_groups
        )
        mutually_exclusive_group_containing_request = next(
            (
                mutually_exclusive_requests_group
                for mutually_exclusive_requests_group in (
                    mutually_exclusive_requests_groups
                )
                if request in mutually_exclusive_requests_group
            ),
            list[ServiceRequest](),
        )
        return all(
            self.current_placement.rejects(other_request)
            for other_request in mutually_exclusive_group_containing_request
            if other_request != request
        )

    def _would_accepting_request_meet_minimum_main_request_ratio(
        self, request: ServiceRequest
    ) -> bool:
        if isinstance(request, MainServiceRequest):
            result = True
        else:
            num_currently_accepted_requests = len(
                self.current_placement.accepted_requests
            )
            num_currently_accepted_main_requests = len(
                self.current_placement.accepted_main_requests
            )
            ratio_of_accepted_main_requests = num_currently_accepted_main_requests / (
                num_currently_accepted_requests + 1
            )
            result = (
                ratio_of_accepted_main_requests
                >= self.current_placement.minimum_ratio_of_main_requests
            )
        return result

    def _find_suitable_nodes_for_request(
        self,
        request: ServiceRequest,
    ) -> Optional[list[NetworkNode]]:
        service = request.requested_service
        service_topology = service.service_topology
        network_topology = self.current_placement.network_topology
        suitable_nodes: Optional[list[NetworkNode]] = [
            service.ingress_node.network_node
        ]
        assert suitable_nodes is not None
        remaining_latency_tolerance = service.maximum_tolerated_latency
        for vnf in service.vnf_chain:
            incoming_virtual_link = next(iter(service_topology.incoming_links(vnf)))
            # Choose a suitable node for the VNF. By shuffling the list before calling
            # next, all suitable nodes have the same probability of being chosen.
            suitable_node = next(
                (
                    node
                    for node in collection_utils.shuffle(network_topology.sorted_nodes)
                    if self._network_node_is_suitable(
                        node,
                        vnf,
                        (
                            incoming_network_link := network_topology.link(
                                suitable_nodes[-1], node
                            )
                        ),
                        incoming_virtual_link,
                        remaining_latency_tolerance,
                    )
                ),
                None,
            )
            # If a suitable node is found, update remaining network resources.
            if suitable_node is not None:
                suitable_nodes.append(suitable_node)
                self._remaining_resources_per_node[
                    suitable_node
                ] = collection_utils.subtract_counters(
                    self._remaining_resources_per_node[suitable_node],
                    vnf.resources_needed_by_resource_type,
                )
                remaining_latency_tolerance -= incoming_network_link.latency
                self._remaining_bandwidth_per_network_link[
                    incoming_network_link
                ] -= incoming_virtual_link.minimum_guaranteed_bandwidth
            else:
                # If no suitable node is found, set suitable nodes to None and
                # break the loop.
                suitable_nodes = None
                break
        else:
            # If all VNFs have been placed, check if the link between the last
            # suitable network node and the egress node is also suitable.
            incoming_network_link = network_topology.link(
                suitable_nodes[-1], service.egress_node.network_node
            )
            suitable_nodes.append(service.egress_node.network_node)
            if not (
                self._incoming_network_link_is_suitable(
                    incoming_network_link,
                    incoming_virtual_link,
                    remaining_latency_tolerance,
                )
            ):
                # If the link is not suitable, set suitable nodes to None.
                suitable_nodes = None
        if suitable_nodes is None:
            # Restore network resources to how they were before attempting to
            # place the request.
            self._remaining_resources_per_node = copy(
                self.current_placement.remaining_resources_per_node
            )
            self._remaining_bandwidth_per_network_link = copy(
                self.current_placement.remaining_bandwidth_per_network_link
            )
        return suitable_nodes

    def _network_node_is_suitable(
        self,
        node: NetworkNode,
        vnf: VNF,
        incoming_network_link: NetworkLink,
        incoming_virtual_link: VirtualLink,
        remaining_latency_tolerance: float,
    ) -> bool:
        return (
            (self._network_node_has_sufficient_resources(node, vnf))
            and self._network_node_is_compatible_with_vnf(node, vnf)
            and self._incoming_network_link_is_suitable(
                incoming_network_link,
                incoming_virtual_link,
                remaining_latency_tolerance,
            )
        )

    def _network_node_has_sufficient_resources(
        self, node: NetworkNode, vnf: VNF
    ) -> bool:
        return all(
            value >= 0
            for value in collection_utils.subtract_counters(
                self._remaining_resources_per_node[node],
                vnf.resources_needed_by_resource_type,
            ).values()
        )

    def _network_node_is_compatible_with_vnf(self, node: NetworkNode, vnf: VNF) -> bool:
        incompatible_nodes = (
            set()
            if self.current_placement.incompatible_nodes_per_vnf is None
            else self.current_placement.incompatible_nodes_per_vnf.get(vnf, set())
        )
        return node not in incompatible_nodes

    def _incoming_network_link_is_suitable(
        self,
        incoming_network_link: NetworkLink,
        incoming_virtual_link: VirtualLink,
        remaining_latency_tolerance: float,
    ) -> bool:
        return (
            incoming_virtual_link.minimum_guaranteed_bandwidth
            <= self._remaining_bandwidth_per_network_link[incoming_network_link]
        ) and (incoming_network_link.latency <= remaining_latency_tolerance)
