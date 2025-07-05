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

def validate_dataset(records, field_rules) -> (str, int):
    """Validate dataset records against field rules and return a report and violation count.
    
    Returns:
        str: The validation report
        int: Number of rule violations found
    """
    # We'll collect the report lines and count violations
    report_lines = []
    total_violations = 0
    violation_details = []  # To collect specific violation messages
    
    report_lines.append("\nValidating dataset...")
    report_lines.append("="*50)
    
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
    
    # Build report and check for rule violations
    for field, stat in stats.items():
        rules = field_rules[field]
        report_lines.append(f"Field: {field}")
        
        allow_null = rules.get('allow_null', True)
        allow_dups = rules.get('allow_duplicates', True)
        
        # Check for null violations
        null_violation = not allow_null and stat['null_like_count'] > 0
        report_lines.append(
            f"  Null-like values: {stat['null_like_count']} (allowed: {allow_null})" + 
            ("   <<< VIOLATION!" if null_violation else "")
        )
        
        # Check for duplicate violations
        dup_violation = not allow_dups and stat['distinct_duplicate_values'] > 0
        report_lines.append(
            f"  Duplicate values: {stat['duplicate_count']} records (distinct: {stat['distinct_duplicate_values']}) (allowed: {allow_dups})" + 
            ("   <<< VIOLATION!" if dup_violation else "")
        )
        
        # Collect violation details
        if null_violation:
            violation_details.append(
                f"Field '{field}' has {stat['null_like_count']} null-like values (not allowed)"
            )
        if dup_violation:
            violation_details.append(
                f"Field '{field}' has {stat['distinct_duplicate_values']} distinct duplicate values (not allowed)"
            )
        
        # Count violations
        total_violations += int(null_violation) + int(dup_violation)
    
    report_lines.append("="*50)
    
    # Include dedicated violations section if any were found
    if violation_details:
        report_lines.append("\n!!! VALIDATION VIOLATIONS DETECTED !!!")
        report_lines.append("-"*50)
        for i, violation in enumerate(violation_details, 1):
            report_lines.append(f"[VIOLATION {i}] {violation}")
        report_lines.append("-"*50)
    else:
        report_lines.append("\nNo validation violations detected")
    
    report = "\n".join(report_lines)
    return report, total_violations
