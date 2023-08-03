import json
from json import JSONEncoder
from typing import Any, Optional

from network import NetworkNode
from service import VNF
from service_request import (
    AlternativeServiceRequest,
    ServiceRequest,
)
from vnf_placement import MultiFlavouredVNFChainPlacement


class CustomJSONEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, MultiFlavouredVNFChainPlacement):
            return self._serialize_multiflavoured_vnf_chain_placement(o)
        return super().default(o)

    def _serialize_multiflavoured_vnf_chain_placement(
        self,
        placement: MultiFlavouredVNFChainPlacement,
    ) -> dict[str, Any]:
        serialization = dict[str, Any]()
        serialization["num_ric"] = len(placement.evaluated_requests)
        serialization["num_ric_main"] = len(placement.evaluated_main_requests)
        serialization["num_ric_sec"] = len(placement.evaluated_alternative_requests)
        serialization["num_ric_serv"] = len(placement.accepted_requests)
        serialization["num_ric_serv_main"] = len(placement.accepted_main_requests)
        serialization["num_ric_serv_sec"] = len(placement.accepted_alternative_requests)
        serialization["obj_val"] = placement.profit
        serialization["requests"] = [
            self._serialize_service_request(
                index,
                request,
                placement.designated_nodes_per_request[request],
            )
            for index, request in enumerate(placement.evaluated_requests)
        ]
        return serialization

    def _serialize_service_request(
        self,
        request_index: int,
        request: ServiceRequest,
        designated_nodes: Optional[list[NetworkNode]],
    ) -> dict[str, Any]:
        serialization = dict[str, Any]()
        serialization["id_richiesta"] = request_index
        serialization["req_type"] = int(isinstance(request, AlternativeServiceRequest))
        serialization["status"] = int(designated_nodes is None)
        serialization["value_y"] = float(designated_nodes is not None)
        serialization["vnfs"] = [
            self._serialize_vnf(index, vnf, designated_nodes[1:-1][index])
            for index, vnf in enumerate(request.requested_service.vnf_chain)
            if designated_nodes is not None
        ]
        return serialization

    def _serialize_vnf(
        self, vnf_index: int, vnf: VNF, designated_node: NetworkNode
    ) -> dict[str, Any]:
        serialization = dict[str, Any]()
        serialization["id_vnf"] = vnf_index
        serialization["position"] = int(designated_node.label)
        serialization["resources"] = [
            vnf.resources_needed_by_resource_type[key]
            for key in sorted(vnf.resources_needed_by_resource_type.keys())
        ]
        serialization["value_y"] = 1.0
        return serialization


class Output:
    def __init__(self, placement: Optional[MultiFlavouredVNFChainPlacement]):
        self.placement = placement

    def to_file(self, output_file_path: str) -> None:
        with open(output_file_path, mode="w", encoding="utf8") as output_file:
            if self.placement is not None:
                json.dump(self.placement, output_file, cls=CustomJSONEncoder, indent=4)
