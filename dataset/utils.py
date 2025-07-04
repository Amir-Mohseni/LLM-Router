import json
import hashlib

def compute_unique_id(record: dict) -> str:
    """Compute the MD5 hash of the record in a canonical JSON representation (without the 'unique_id' field)."""
    # We assume the record does not have the key 'unique_id'
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(canonical.encode('utf-8')).hexdigest()
