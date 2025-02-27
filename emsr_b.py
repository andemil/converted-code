import numpy as np
from scipy.stats import poisson

def emsr_b(fares=None, cancel_prob=0.20, capacity=100, arrival_rate=1.4, 
           time_horizon=100, prob_class=None):
    """
    Compute EMSR-b protection levels and booking limits for a set of fare classes.
    
    Parameters:
    -----------
    fares : array-like
        Fares for each class (should be in increasing order)
    cancel_prob : float
        Probability of cancellation
    capacity : float
        Aircraft seat capacity
    arrival_rate : float
        Arrival rate per day
    time_horizon : float
        Time horizon in days
    prob_class : array-like
        Probability of requests for each fare class
        
    Returns:
    --------
    fares : array
        Fares for each class
    booking_limits : array
        Booking limits for each class
    """
    # Default values if parameters are not provided
    if fares is None:
        fares = np.array([101, 127, 153, 179, 293, 419])
    
    if prob_class is None:
        prob_class = np.array([0.3, 0.3, 0.13, 0.13, 0.09, 0.05])
    
    # Validate input dimensions
    num_classes = len(fares)
    if len(prob_class) != num_classes:
        raise ValueError("Probability array must have same length as fares array")
    
    # Verify fares are in increasing order
    if not np.all(np.diff(fares) >= 0):
        print("Warning: Fares should ideally be in increasing order for EMSR-b")
    
    # Compute expected (mean) demand for each fare class
    mean_demand = arrival_rate * time_horizon * prob_class
    
    # Compute aggregate demands and weighted average fares (clustering from highest fare)
    aggregate_demand = np.zeros(num_classes)
    aggregate_fare = np.zeros(num_classes)
    total_demand = 0
    total_weighted_fare = 0
    
    # Process in reverse order (from highest to lowest fare)
    for j in range(num_classes - 1, -1, -1):
        total_demand += mean_demand[j]
        total_weighted_fare += fares[j] * mean_demand[j]
        aggregate_demand[j] = total_demand
        aggregate_fare[j] = total_weighted_fare / total_demand
    
    protection_levels = np.zeros(num_classes)
    protection_levels[0] = 0  # for the highest fare class, protection is 0
    
    # Compute protection levels for lower fare classes
    for i in range(num_classes - 1):
        # Calculate the fraction used in the inverse Poisson calculation
        fraction = (aggregate_fare[i+1] - fares[i]) / aggregate_fare[i+1]
        if fraction <= 0:  # Edge case check
            protection_levels[i+1] = 0
        else:
            # Use the Poisson inverse (percent point function)
            protection_levels[i+1] = poisson.ppf(fraction, aggregate_demand[i+1])
    
    # Adjust capacity for cancellations
    C_adjusted = capacity / (1 - cancel_prob)
    
    # Compute booking limits for each fare class
    booking_limits = np.zeros(num_classes)
    for i in range(num_classes - 1):
        booking_limits[i] = max(C_adjusted - protection_levels[i+1], 0)
    booking_limits[num_classes - 1] = C_adjusted
    
    # Print results
    print("\nEMSR-b Results:")
    print("Fare Class   | Fare  | Booking Limit")
    print("-------------|-------|-------------")
    for i in range(num_classes):
        print(f"Class {i+1:6} | {fares[i]:5.0f} | {booking_limits[i]:8.2f}")
    
    return fares, booking_limits
