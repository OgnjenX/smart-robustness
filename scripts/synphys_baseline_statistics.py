"""Pure pulse-weighted statistics; no metadata queries or eligibility promotion.

The source pipeline uses nanmean over contributing pulse responses, not unique
recordings. SQLite nulls cannot silently be reinterpreted as omitted NaNs.
"""

from __future__ import annotations

import math
import sys


def pulse_baseline_statistics(values):
    """Retain missingness while separating a source mean from complete coverage."""
    values = list(values)
    if len(values) > 100000:
        raise ValueError("original pulse-ID count bound exceeded")
    finite, nan_count, null_count, infinity_count, invalid_count = [], 0, 0, 0, 0
    for value in values:
        if value is None:
            null_count += 1
        elif type(value) not in (int, float):
            invalid_count += 1
        elif math.isnan(value):
            nan_count += 1
        elif not math.isfinite(value):
            infinity_count += 1
        else:
            finite.append(float(value))
    # fsum provides an independent high-accuracy reference, not historical
    # NumPy summation parity. No nominal voltage or null value is inserted.
    mean = math.fsum(finite) / len(finite) if finite else None
    deviation = (math.sqrt(math.fsum((value - mean) ** 2 for value in finite) / len(finite))
                 if finite else None)
    source_mean_unambiguous = bool(finite) and not (null_count or infinity_count or invalid_count)
    return {
        "pulse_count": len(values), "finite_count": len(finite), "nan_count": nan_count,
        "null_count": null_count, "infinity_count": infinity_count, "invalid_type_count": invalid_count,
        "finite_only_mean_volts": mean, "finite_only_population_std_volts": deviation,
        "finite_only_min_volts": min(finite) if finite else None,
        "finite_only_max_volts": max(finite) if finite else None,
        "source_nanmean_unambiguous": source_mean_unambiguous,
        "source_nanmean_reference_volts": mean if source_mean_unambiguous else None,
        "complete_finite_baseline_coverage": bool(values) and len(finite) == len(values),
        "biological_eligibility_established": False,
    }


def numerical_mean_bound(values):
    """Conservative binary64 mean discrepancy bound, not a biological tolerance.

    Bound two summation orders and division rounding. This is only defined
    for nonempty finite values within the already bounded pulse-ID extent.
    """
    values = list(values)
    if (not values or len(values) > 100000
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
        raise ValueError("bounded finite baseline values required")
    count = len(values)
    epsilon = sys.float_info.epsilon
    gamma = count * epsilon / (1 - count * epsilon)
    mean_abs = math.fsum(abs(value) for value in values) / count
    mean = math.fsum(values) / count
    return 2 * gamma * mean_abs + 2 * math.ulp(mean)
