import dataclasses
import json
import hashlib
import numpy as np


def json_default(data):
    try:
        return data.__name__
    except:
        return type(data).__name__


def serialize_dataclass(instance) -> dict:
    if dataclasses.is_dataclass(instance):
        instance_dict = instance.__dict__
        if hasattr(instance, "custom_dict"):
            instance_dict = instance.custom_dict()
        return {
            "__class__": type(instance).__name__,
            "__data__": {k: serialize_dataclass(v) for k, v in instance_dict.items()},
        }
    elif isinstance(instance, list):
        return [serialize_dataclass(i) for i in instance]
    elif isinstance(instance, np.ndarray):
        return serialize_dataclass(instance.tolist())
    elif isinstance(instance, dict):
        return {k: serialize_dataclass(v) for k, v in instance.items()}
    else:
        return instance


def json_dumps_dataclass_str(data_class, indent=None):
    return json.dumps(
        serialize_dataclass(data_class),
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=(",", ":"),
    )


def json_dumps_dataclass(data_class):
    return json_dumps_dataclass_str(data_class).encode("utf-8")


def stable_hash(data_class):
    json_str = json_dumps_dataclass(data_class)
    return int(hashlib.md5(json_str).hexdigest(), 16) & ((1 << 64) - 1)


def stable_hash_small(data_class):
    return stable_hash_str(data_class)


def i64_to_hex(i64_val: int) -> str:
    return f"{i64_val:016x}"


def hex_to_i64(hex_str: str) -> int:
    return int(hex_str, 16)


def stable_hash_str(data_class):
    return i64_to_hex(stable_hash(data_class))
