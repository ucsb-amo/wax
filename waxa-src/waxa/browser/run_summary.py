from dataclasses import dataclass, field


@dataclass
class RunSummary:
    run_id: int
    experiment_name: str
    experiment_filepath: str
    run_date_str: str
    run_datetime_str: str
    filepath: str
    xvarnames: list[str]
    xvardims: tuple[int, ...]
    data_container_keys: list[str]
    has_scope_data: bool
    n_repeats: int = 1
    has_lite: bool = False
    xvar_details: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    comment: str = ""

    def to_cache_dict(self):
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "experiment_filepath": self.experiment_filepath,
            "run_date_str": self.run_date_str,
            "run_datetime_str": self.run_datetime_str,
            "filepath": self.filepath,
            "xvarnames": list(self.xvarnames),
            "xvardims": list(self.xvardims),
            "data_container_keys": list(self.data_container_keys),
            "has_scope_data": self.has_scope_data,
            "n_repeats": int(self.n_repeats),
            "has_lite": self.has_lite,
            "tags": list(self.tags),
            "comment": self.comment,
        }

    @classmethod
    def from_cache_dict(cls, payload: dict):
        return cls(
            run_id=int(payload["run_id"]),
            experiment_name=payload.get("experiment_name", ""),
            experiment_filepath=payload.get("experiment_filepath", ""),
            run_date_str=payload.get("run_date_str", ""),
            run_datetime_str=payload.get("run_datetime_str", ""),
            filepath=payload.get("filepath", ""),
            xvarnames=list(payload.get("xvarnames", [])),
            xvardims=tuple(int(value) for value in payload.get("xvardims", [])),
            data_container_keys=list(payload.get("data_container_keys", [])),
            has_scope_data=bool(payload.get("has_scope_data", False)),
            n_repeats=int(payload.get("n_repeats", 1) or 1),
            has_lite=bool(payload.get("has_lite", False)),
            tags=list(payload.get("tags", [])),
            comment=payload.get("comment", ""),
        )
