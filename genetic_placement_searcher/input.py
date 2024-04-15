from collections import Counter
from itertools import pairwise
import re
from typing import Optional, Sequence, cast

import collection_utils
import function_utils
import string_utils
from network import (
    NetworkTopology,
    NetworkLink,
    NetworkNode,
    NetworkResource,
)
from service import (
    VNF,
    Service,
    ServiceEndpoint,
    VirtualLink,
    VirtualNode,
)
from service_request import (
    AlternativeServiceRequest,
    MainServiceRequest,
    ServiceRequest,
)


class InputParser:
    def __init__(self, input_file_path: str, encoding: Optional[str] = None) -> None:
        with open(input_file_path, encoding=encoding) as input_file:
            file_content = input_file.read()
        self.num_resources: int = 0
        self.num_vnfs_per_request: int = 0
        self.node_indexes = list[int]()
        self.minimum_ratio_of_main_requests: float = 0
        self.main_request_indexes = list[int]()
        self.alt_request_indexes_per_main_request_index = dict[int, list[int]]()
        self.request_indexes = list[int]()
        self.vnf_types = list[int]()
        self.network_link_latency_matrix = list[list[float]]()
        self.network_link_bandwidth_matrix = list[list[float]]()
        self.resource_unit_costs_per_node = dict[int, tuple[float, ...]]()
        self.network_link_bandwidth_unit_cost_matrix = list[list[float]]()
        self.incompatible_nodes_per_request = dict[int, dict[int, set[int]]]()
        self.service_endpoint_node_indexes_per_request = dict[int, tuple[int, int]]()
        self.latency_per_request = dict[int, float]()
        self.revenue_per_request = dict[int, float]()
        self.vnf_types_per_request = dict[int, list[int]]()
        self.minimum_bandwidths_per_request = dict[int, list[float]]()
        self.resources_needed_per_request = dict[int, list[tuple[int, ...]]]()
        self.resource_capacities_per_node = dict[int, tuple[int, ...]]()
        self._parse_file_content(file_content)

    def _parse_file_content(self, file_content: str) -> None:
        try:
            formatted_file_content = self._format_file_content_for_parsing(file_content)
            rows_to_parse = string_utils.split_string_into_matrix(
                formatted_file_content
            )
            self._parse_general_information_header(rows_to_parse.pop(0))
            self._parse_minimum_ratio_of_main_requests(rows_to_parse.pop(0))
            self._parse_request_indexes([rows_to_parse.pop(0) for _ in range(3)])
            self._parse_vnf_types(rows_to_parse.pop(0))
            self._parse_link_latency_matrix(
                [rows_to_parse.pop(0) for _ in self.node_indexes]
            )
            self._parse_link_bandwidth_matrix(
                [rows_to_parse.pop(0) for _ in self.node_indexes]
            )
            self._parse_node_cost_matrix(
                [rows_to_parse.pop(0) for _ in self.node_indexes]
            )
            self._parse_link_bandwidth_unit_cost_matrix(
                [rows_to_parse.pop(0) for _ in self.node_indexes]
            )
            self._parse_incompatibility_matrix(
                [
                    rows_to_parse.pop(0)
                    for _ in range(
                        len(self.request_indexes) * self.num_vnfs_per_request
                    )
                ]
            )
            self._parse_request_information(
                [rows_to_parse.pop(0) for _ in self.request_indexes]
            )
            self._parse_vnf_chain_information(
                [rows_to_parse.pop(0) for _ in self.request_indexes]
            )
            self._parse_vnf_resource_requirements_matrix(
                [rows_to_parse.pop(0) for _ in self.request_indexes]
            )
            self._parse_node_capacity_matrix(
                [rows_to_parse.pop(0) for _ in self.node_indexes]
            )
        except IndexError as exc:
            raise ValueError("Unexpected EOF. File may be too short.") from exc

    def _format_file_content_for_parsing(self, file_content: str) -> str:
        formatting_functions = [
            string_utils.remove_duplicate_whitespace_characters,
            self._remove_comments,
            self._strip_whitespace_line_by_line,
            self._remove_blank_lines,
        ]
        formatted_file_content = function_utils.compose(
            formatting_functions, file_content
        )
        return formatted_file_content

    def _remove_comments(self, file_content: str) -> str:
        modified_file_content = re.sub("[#;].*?\\n", "", file_content)
        return modified_file_content

    def _strip_whitespace_line_by_line(self, file_content: str) -> str:
        modified_file_content = str().join(
            [line.strip() + "\n" for line in file_content.splitlines()]
        )
        return modified_file_content

    def _remove_blank_lines(self, file_content: str) -> str:
        modified_file_content = re.sub("^\\n+", "", file_content)
        modified_file_content = re.sub("\\n+", "\n", modified_file_content)
        return modified_file_content

    def _parse_general_information_header(self, row_to_parse: list[str]) -> None:
        (
            num_nodes,
            _,
            _,
            self.num_resources,
            _,
            self.num_vnfs_per_request,
        ) = [int(elem) for elem in row_to_parse]
        self.node_indexes = list(range(num_nodes))

    def _parse_minimum_ratio_of_main_requests(self, row_to_parse: list[str]) -> None:
        self.minimum_ratio_of_main_requests = float(row_to_parse[0])

    def _parse_request_indexes(self, rows_to_parse: list[list[str]]) -> None:
        self.main_request_indexes = [int(elem) for elem in rows_to_parse[1]]
        alt_request_indexes = [int(elem) for elem in rows_to_parse[2]]
        num_alt_requests_per_main_request_index = [
            int(elem) for elem in rows_to_parse[0]
        ]
        self.alt_request_indexes_per_main_request_index = dict(
            zip(
                self.main_request_indexes,
                collection_utils.chunk(
                    alt_request_indexes, sizes=num_alt_requests_per_main_request_index
                ),
            )
        )
        self.request_indexes = collection_utils.chain_lists(
            [main] + alts
            for main, alts in self.alt_request_indexes_per_main_request_index.items()
        )

    def _parse_vnf_types(self, row_to_parse: list[str]) -> None:
        self.vnf_types = [int(elem) for elem in row_to_parse]

    def _parse_link_latency_matrix(self, rows_to_parse: list[list[str]]) -> None:
        self.network_link_latency_matrix = [
            [float(elem) for elem in row] for row in rows_to_parse
        ]

    def _parse_link_bandwidth_matrix(self, rows_to_parse: list[list[str]]) -> None:
        self.network_link_bandwidth_matrix = [
            [float(elem) for elem in row] for row in rows_to_parse
        ]

    def _parse_node_cost_matrix(self, rows_to_parse: list[list[str]]) -> None:
        self.resource_unit_costs_per_node = {
            index: tuple(float(elem) for elem in row)
            for index, row in enumerate(rows_to_parse)
        }

    def _parse_link_bandwidth_unit_cost_matrix(
        self, rows_to_parse: list[list[str]]
    ) -> None:
        self.network_link_bandwidth_unit_cost_matrix = [
            [float(elem) for elem in row] for row in rows_to_parse
        ]

    def _parse_incompatibility_matrix(self, rows_to_parse: list[list[str]]) -> None:
        rows_by_request = collection_utils.chunk(
            rows_to_parse, self.num_vnfs_per_request
        )
        self.incompatible_nodes_per_request = {
            request_index: {
                vnf_index: {int(elem) - 1 for elem in row[2:] if elem.isnumeric()}
                for vnf_index, row in enumerate(rows_by_vnf_index)
            }
            for request_index, rows_by_vnf_index in enumerate(rows_by_request)
        }

    def _parse_request_information(self, rows_to_parse: list[list[str]]) -> None:
        self.service_endpoint_node_indexes_per_request = {
            int(float(row[0])): (
                int(float(row[1])),
                int(float(row[2])),
            )
            for row in rows_to_parse
        }
        self.latency_per_request = {
            int(float(row[0])): float(row[3]) for row in rows_to_parse
        }
        self.revenue_per_request = {
            int(float(row[0])): float(row[5]) for row in rows_to_parse
        }

    def _parse_vnf_chain_information(self, rows_to_parse: list[list[str]]) -> None:
        self.vnf_types_per_request = {
            int(float(row[0])): [
                int(float(elem)) for elem in row[2 : 2 + int(float(row[1]))]
            ]
            for row in rows_to_parse
        }
        self.minimum_bandwidths_per_request = {
            int(float(row[0])): [float(elem) for elem in row[2 + int(float(row[1])) :]]
            for row in rows_to_parse
        }

    def _parse_vnf_resource_requirements_matrix(
        self, rows_to_parse: list[list[str]]
    ) -> None:
        self.resources_needed_per_request = {
            request_index: [
                tuple(int(float(elem)) for elem in resources_needed_per_resource_type)
                for resources_needed_per_resource_type in collection_utils.chunk(
                    row[1:], int(len(row[1:]) / int(float(row[0])))
                )
            ]
            for request_index, row in enumerate(rows_to_parse)
        }

    def _parse_node_capacity_matrix(self, rows_to_parse: list[list[str]]) -> None:
        self.resource_capacities_per_node = {
            node_index: tuple(int(elem) for elem in row)
            for node_index, row in enumerate(rows_to_parse)
        }


class Input:
    def __init__(
        self,
        requests: list[ServiceRequest],
        network: NetworkTopology,
        minimum_ratio_of_main_requests: float,
        incompatible_nodes_per_vnf: Optional[dict[VNF, set[NetworkNode]]],
    ):
        self.requests = requests
        self.network = network
        self.minimum_ratio_of_main_requests = minimum_ratio_of_main_requests
        self.incompatible_nodes_per_vnf = incompatible_nodes_per_vnf

    @classmethod
    def from_file(cls, input_file_path: str) -> "Input":
        input_parser = InputParser(input_file_path, encoding="utf-8")
        resources = cls._create_network_resources(input_parser.num_resources)
        network = cls._create_network_topology(
            input_parser.node_indexes,
            resources,
            input_parser.resource_capacities_per_node,
            input_parser.resource_unit_costs_per_node,
            input_parser.network_link_bandwidth_matrix,
            input_parser.network_link_bandwidth_unit_cost_matrix,
            input_parser.network_link_latency_matrix,
        )
        requests = cls._create_requests(
            input_parser.request_indexes,
            input_parser.alt_request_indexes_per_main_request_index,
            network.sorted_nodes,
            input_parser.service_endpoint_node_indexes_per_request,
            input_parser.latency_per_request,
            input_parser.num_vnfs_per_request,
            resources,
            input_parser.resources_needed_per_request,
            input_parser.minimum_bandwidths_per_request,
            input_parser.revenue_per_request,
        )
        minimum_ratio_of_main_requests = input_parser.minimum_ratio_of_main_requests
        incompatible_nodes_per_vnf = cls._create_incompatible_nodes_per_vnf(
            requests, network.sorted_nodes, input_parser.incompatible_nodes_per_request
        )
        return Input(
            requests,
            network,
            minimum_ratio_of_main_requests,
            incompatible_nodes_per_vnf,
        )

    @classmethod
    def _create_network_resources(cls, num_resources: int) -> list[NetworkResource]:
        resources = [NetworkResource(str(i)) for i in range(num_resources)]
        return resources

    @classmethod
    def _create_network_topology(
        cls,
        node_indexes: list[int],
        resources: list[NetworkResource],
        resource_capacities_per_node: dict[int, tuple[int, ...]],
        resource_unit_costs_per_node: dict[int, tuple[float, ...]],
        link_bandwidth_matrix: list[list[float]],
        link_bandwidth_unit_cost_matrix: list[list[float]],
        link_latency_matrix: list[list[float]],
    ) -> NetworkTopology:
        nodes = cls._create_nodes(
            node_indexes,
            resources,
            resource_capacities_per_node,
            resource_unit_costs_per_node,
        )
        links = cls._create_network_links(
            nodes,
            link_bandwidth_matrix,
            link_bandwidth_unit_cost_matrix,
            link_latency_matrix,
        )
        return NetworkTopology(set(nodes), links)

    @classmethod
    def _create_nodes(
        cls,
        node_indexes: list[int],
        resources: list[NetworkResource],
        resource_capacities_per_node: dict[int, tuple[int, ...]],
        resource_unit_costs_per_node: dict[int, tuple[float, ...]],
    ) -> list[NetworkNode]:
        nodes = []
        for index in node_indexes:
            label = str(index)
            capacity_per_resource_type = Counter(
                dict(zip(resources, resource_capacities_per_node[index]))
            )
            unit_cost_per_resource_type = dict(
                zip(resources, resource_unit_costs_per_node[index])
            )
            node = NetworkNode(
                label, capacity_per_resource_type, unit_cost_per_resource_type
            )
            nodes.append(node)
        return nodes

    @classmethod
    def _create_network_links(
        cls,
        nodes: list[NetworkNode],
        link_bandwidth_matrix: list[list[float]],
        link_bandwidth_unit_cost_matrix: list[list[float]],
        link_latency_matrix: list[list[float]],
    ) -> set[NetworkLink]:
        network_links = set()
        for tail_index, tail in enumerate(nodes):
            for head_index, head in enumerate(nodes):
                latency = link_latency_matrix[tail_index][head_index]
                bandwidth = link_bandwidth_matrix[tail_index][head_index]
                bandwidth_unit_cost = link_bandwidth_unit_cost_matrix[tail_index][
                    head_index
                ]
                link = NetworkLink(tail, head, latency, bandwidth, bandwidth_unit_cost)
                network_links.add(link)
        return network_links

    @classmethod
    def _create_requests(
        cls,
        request_indexes: list[int],
        alt_request_indexes_per_main_request_index: dict[int, list[int]],
        nodes: list[NetworkNode],
        service_endpoint_node_indexes_per_request: dict[int, tuple[int, int]],
        latency_per_request: dict[int, float],
        num_vnfs_per_request: int,
        resources: list[NetworkResource],
        resources_needed_per_request: dict[int, list[tuple[int, ...]]],
        minimum_bandwidths_per_request: dict[int, list[float]],
        revenue_per_request: dict[int, float],
    ) -> list[ServiceRequest]:
        service_per_request = cls._create_services(
            request_indexes,
            nodes,
            service_endpoint_node_indexes_per_request,
            latency_per_request,
            num_vnfs_per_request,
            resources,
            resources_needed_per_request,
            minimum_bandwidths_per_request,
        )
        main_requests = cls._create_main_requests(
            alt_request_indexes_per_main_request_index,
            service_per_request,
            revenue_per_request,
        )
        requests = cast(
            list[ServiceRequest],
            collection_utils.chain_lists(
                [main_request] + main_request.alternative_requests
                for main_request in main_requests
            ),
        )
        return requests

    @classmethod
    def _create_main_requests(
        cls,
        alt_request_indexes_per_main_request_index: dict[int, list[int]],
        service_per_request: dict[int, Service],
        revenue_per_request: dict[int, float],
    ) -> list[MainServiceRequest]:
        main_requests = []
        for (
            main_request_index,
            alt_request_indexes,
        ) in alt_request_indexes_per_main_request_index.items():
            alt_requests = cls._create_alternative_requests(
                alt_request_indexes, service_per_request, revenue_per_request
            )
            main_request = MainServiceRequest(
                service_per_request[main_request_index],
                revenue_per_request[main_request_index],
                alt_requests,
            )
            main_requests.append(main_request)
        return main_requests

    @classmethod
    def _create_alternative_requests(
        cls,
        alt_request_indexes: list[int],
        service_per_request: dict[int, Service],
        revenue_per_request: dict[int, float],
    ) -> list[AlternativeServiceRequest]:
        alt_requests = [
            AlternativeServiceRequest(
                service_per_request[alt_request_index],
                revenue_per_request[alt_request_index],
            )
            for alt_request_index in alt_request_indexes
        ]
        return alt_requests

    @classmethod
    def _create_services(
        cls,
        request_indexes: list[int],
        nodes: list[NetworkNode],
        service_endpoint_node_indexes_per_request: dict[int, tuple[int, int]],
        latency_per_request: dict[int, float],
        num_vnfs_per_request: int,
        resources: list[NetworkResource],
        resources_needed_per_request: dict[int, list[tuple[int, ...]]],
        minimum_bandwidths_per_request: dict[int, list[float]],
    ) -> dict[int, Service]:
        service_per_request = {
            request_index: cls._create_service(
                nodes,
                service_endpoint_node_indexes_per_request[request_index],
                latency_per_request[request_index],
                num_vnfs_per_request,
                resources,
                resources_needed_per_request[request_index],
                minimum_bandwidths_per_request[request_index],
            )
            for request_index in request_indexes
        }
        return service_per_request

    @classmethod
    def _create_service(
        cls,
        nodes: list[NetworkNode],
        network_endpoints: tuple[int, int],
        maximum_tolerated_latency: float,
        num_vnfs: int,
        resources: list[NetworkResource],
        resources_needed_per_vnf: list[tuple[int, ...]],
        minimum_bandwidth_per_virtual_link: list[float],
    ) -> Service:
        ingress_node = ServiceEndpoint(nodes[network_endpoints[0]])
        egress_node = ServiceEndpoint(nodes[network_endpoints[1]])
        vnf_chain = cls._create_vnf_chain(num_vnfs, resources, resources_needed_per_vnf)
        virtual_nodes = [ingress_node] + vnf_chain + [egress_node]
        virtual_links = cls._create_virtual_links(
            virtual_nodes, minimum_bandwidth_per_virtual_link
        )
        return Service(
            ingress_node,
            egress_node,
            vnf_chain,
            virtual_links,
            maximum_tolerated_latency,
        )

    @classmethod
    def _create_vnf_chain(
        cls,
        num_vnfs: int,
        resources: list[NetworkResource],
        resources_needed_per_vnf: list[tuple[int, ...]],
    ) -> list[VNF]:
        vnf_chain = []
        for vnf_index in range(num_vnfs):
            resources_needed_by_resource_type = Counter(
                dict(zip(resources, resources_needed_per_vnf[vnf_index]))
            )
            vnf = VNF(resources_needed_by_resource_type)
            vnf_chain.append(vnf)
        return vnf_chain

    @classmethod
    def _create_virtual_links(
        cls,
        virtual_nodes: Sequence[VirtualNode],
        minimum_bandwidth_per_virtual_link: list[float],
    ) -> set[VirtualLink]:
        virtual_links = set()
        for link_index, (tail, head) in enumerate(pairwise(virtual_nodes)):
            minimum_guaranteed_bandwidth = minimum_bandwidth_per_virtual_link[
                link_index
            ]
            virtual_link = VirtualLink(tail, head, minimum_guaranteed_bandwidth)
            virtual_links.add(virtual_link)
        return virtual_links

    @classmethod
    def _create_incompatible_nodes_per_vnf(
        cls,
        requests: Sequence[ServiceRequest],
        nodes: list[NetworkNode],
        incompatible_node_indexes_per_request_index: dict[int, dict[int, set[int]]],
    ) -> Optional[dict[VNF, set[NetworkNode]]]:
        incompatible_nodes_per_vnf = {
            requests[request_index].requested_service.vnf_chain[vnf_index]: {
                nodes[node_index] for node_index in node_indexes
            }
            for request_index, node_indexes_per_vnf_index in (
                incompatible_node_indexes_per_request_index.items()
            )
            for vnf_index, node_indexes in node_indexes_per_vnf_index.items()
        }
        return incompatible_nodes_per_vnf or None
