"""
JSON serialization helpers that convert numpy/pandas scalars and other
non-JSON-serializable types to native Python types so json.dumps() never fails.
"""
import json


def json_safe(obj):
    """
    Return a copy of obj with all values converted to JSON-serializable types.
    Recursively processes dicts and lists; converts numpy/pandas scalars via .item()
    or str() so json.dumps() never raises TypeError.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        return str(obj)


def dumps_safe(obj):
    """
    Serialize obj to a JSON string, first converting to JSON-serializable types.
    Use this whenever obj may contain numpy/pandas scalars or other non-standard types.
    """
    return json.dumps(json_safe(obj))
