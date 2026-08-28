from enum import Enum

class RuntimeMode(str, Enum):
    TRAIN = "TRAIN"
    WORK_OFFLINE = "WORK-OFFLINE"
    WORK_ONLINE = "WORK-ONLINE"
    OUT_SERVICE = "OUT_SERVICE"

    @classmethod
    def parse(cls, value: str) -> "RuntimeMode":
        normalized = value.strip().upper().replace("_", "-")
        if normalized in {"TRAIN", "TRAINING"}:
            return cls.TRAIN
        if normalized in {"WORK-OFFLINE", "OFFLINE", "TEST", "EVALUATE", "EVAL"}:
            return cls.WORK_OFFLINE
        if normalized in {"WORK-ONLINE", "ONLINE"}:
            return cls.WORK_ONLINE
        if normalized in {"OUT-SERVICE", "OUT_SERVICE", "OUTOFSERVICE", "OUT-OF-SERVICE"}:
            return cls.OUT_SERVICE
        raise ValueError(f"Unsupported runtime mode: {value}")
