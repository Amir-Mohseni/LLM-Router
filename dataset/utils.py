import json
import hashlib
from collections import defaultdict

def compute_unique_id(record: dict) -> str:
    """Compute the MD5 hash of the record in a canonical JSON representation (without the 'unique_id' field)."""
    # We assume the record does not have the key 'unique_id'
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(canonical.encode('utf-8')).hexdigest()

def is_null_like(value):
    """Check if a value is null-like: None, empty string, empty list, or empty dict."""
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False

def validate_dataset(records, field_rules):
    """Validate dataset records against field rules and report statistics."""
    print("\nValidating dataset...")
    print("="*50)
    
    # Collect statistics per field
    stats = {}
    for field in field_rules:
        values = [record.get(field) for record in records]
        
        # Count null-like values
        null_like_count = sum(1 for v in values if is_null_like(v))
        
        # Count duplicates (only for non-null-like values)
        non_null_values = [v for v in values if not is_null_like(v)]
        freq = defaultdict(int)
        for v in non_null_values:
            # Make lists hashable by converting to tuple
            if isinstance(v, list):
                v = tuple(v)
            freq[v] += 1
        
        duplicate_count = sum(count for count in freq.values() if count > 1)
        distinct_duplicate_values = sum(1 for count in freq.values() if count > 1)
        
        stats[field] = {
            'null_like_count': null_like_count,
            'duplicate_count': duplicate_count,
            'distinct_duplicate_values': distinct_duplicate_values
        }
    
    # Print report
    for field, stat in stats.items():
        rules = field_rules[field]
        print(f"Field: {field}")
        print(f"  Null-like values: {stat['null_like_count']} (allowed: {rules.get('allow_null', True)})")
        print(f"  Duplicate values: {stat['duplicate_count']} records (distinct: {stat['distinct_duplicate_values']}) (allowed: {rules.get('allow_duplicates', True)})")
    
    print("="*50)
