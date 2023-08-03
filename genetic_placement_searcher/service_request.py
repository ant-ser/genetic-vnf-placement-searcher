from abc import ABC, abstractmethod
from functools import total_ordering
from typing import Optional

from service import Service


@total_ordering
class ServiceRequest(ABC):
    @abstractmethod
    def __init__(self, requested_service: Service, revenue: float):
        self.requested_service = requested_service
        self.revenue = revenue

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ServiceRequest):
            return False
        return (
            self.requested_service == other.requested_service
            and self.revenue == other.revenue
        )

    def __hash__(self) -> int:
        return hash((self.requested_service, self.revenue))

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ServiceRequest):
            return False
        return self.revenue < other.revenue


class AlternativeServiceRequest(ServiceRequest):
    def __init__(self, requested_service: Service, revenue: float):
        super().__init__(requested_service, revenue)


class MainServiceRequest(ServiceRequest):
    def __init__(
        self,
        requested_service: Service,
        revenue: float,
        alternative_requests: Optional[list[AlternativeServiceRequest]] = None,
    ):
        super().__init__(requested_service, revenue)
        self.alternative_requests = alternative_requests or []
