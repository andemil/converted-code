import numpy as np
from scipy.stats import poisson

def emsr_b():
    """
    Implementation of EMSR-b from the example in the MATLAB code.
    Prints the final booking limits for each fare class.
    """
    # Example data
    fares = np.array([101, 127, 153, 179, 293, 419], dtype=float)  # ascending order
    cancel_prob = 0.20
    C = 100  # seat capacity
    arrival_rate = 1.4
    time_horizon = 100
    prob_class = np.array([0.3, 0.3, 0.13, 0.13, 0.09, 0.05], dtype=float)

    num_classes = len(fares)

    # Compute expected demand (mean_demand)
    mean_demand = arrival_rate * time_horizon * prob_class

    # Compute EMSR-b protection levels
    protection_levels = np.zeros(num_classes, dtype=float)
    aggregate_demand = np.zeros(num_classes, dtype=float)
    aggregate_fare = np.zeros(num_classes, dtype=float)

    total_demand = 0.0
    total_weighted_fare = 0.0

    # We do the aggregation in reverse order (like the MATLAB code),
    # but keep track in arrays for clarity.
    # i goes from 1..num_classes in MATLAB, we do reversed indexing in Python:
    # j = num_classes - i + 1
    for i in range(num_classes):
        j = num_classes - i - 1  # 0-based
        total_demand += mean_demand[j]
        aggregate_demand[j] = total_demand
        total_weighted_fare += fares[j] * mean_demand[j]
        aggregate_fare[j] = total_weighted_fare / total_demand

    # Now compute protection levels
    # We skip i=0 because there's no protection for the highest class in the original code
    # i in [1..num_classes-1]
    for i in range(1, num_classes):
        # fraction = (aggregate_fare(i+1) - fares(i)) / aggregate_fare(i+1)
        # but in code: i+1 => we must be mindful of indexing
        fraction = (aggregate_fare[i] - fares[i-1]) / aggregate_fare[i]
        # protection_levels(i+1) = poissinv(fraction, aggregate_demand(i+1))
        protection_levels[i] = poisson.ppf(fraction, aggregate_demand[i])

    protection_levels[0] = 0

    # Adjust capacity for cancellations: C/(1 - cancel_prob)
    effective_capacity = C / (1 - cancel_prob)

    booking_limits = np.zeros(num_classes, dtype=float)
    for i in range(num_classes - 1):
        booking_limits[i] = max(effective_capacity - protection_levels[i+1], 0)
    booking_limits[num_classes - 1] = effective_capacity

    print("Fare Class | Booking Limit")
    for i in range(num_classes):
        print(f"{fares[i]:9.2f} | {booking_limits[i]:.2f}")

    return booking_limits

if __name__ == "__main__":
    emsr_b()
