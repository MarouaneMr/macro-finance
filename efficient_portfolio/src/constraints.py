def apply_long_only_constraint(weights):
    """
    Enforce long-only constraint.
    """
    return all(weights >= 0)

def apply_weight_bounds(weights, lower, upper):
    """
    Enforce weight bounds [lower, upper].
    """
    return all((weights >= lower) & (weights <= upper))