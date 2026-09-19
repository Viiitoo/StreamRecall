"""Versioned evaluation records shared by resource and metric writers."""

from __future__ import annotations

from dataclasses import dataclass

from streamtimelens.protocol.types import PersistentRecord, _finite


@dataclass(frozen=True)
class ResourceRecord(PersistentRecord):
    record_type = "resource_record"
    component: str
    wall_s: float
    cpu_s: float
    cuda_s: float | None
    status: str
    bytes_in: int = 0
    bytes_out: int = 0

    def __post_init__(self) -> None:
        if not self.component or not self.status or self.bytes_in < 0 or self.bytes_out < 0:
            raise ValueError("invalid resource record")
        _finite("wall_s", self.wall_s, non_negative=True)
        _finite("cpu_s", self.cpu_s, non_negative=True)
        if self.cuda_s is not None:
            _finite("cuda_s", self.cuda_s, non_negative=True)

    @classmethod
    def from_dict(cls, data: dict) -> "ResourceRecord":
        return cls(**cls._clean(data))
