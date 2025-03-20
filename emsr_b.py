import numpy as np
from davn_utils import extract_leg_fare_classes

def calculate_emsr_b(fares, mean_demands, probabilities, capacity):
    """
    Implements the EMSR-b algorithm for revenue management.
    
    Parameters:
    fares -- array of fares in descending order (highest fare first)
    mean_demands -- array of mean demands corresponding to the fares
    probabilities -- array of probabilities for each fare class
    capacity -- total capacity available
    
    Returns:
    protection_levels -- array of protection levels
    booking_limits -- array of booking limits
    expected_revenue -- expected revenue from this leg
    """
    # Number of fare classes
    n = len(fares)
    
    # Sort fares, demands, and probabilities in descending order of fares
    sorted_indices = np.argsort(fares)[::-1]  # descending
    fares = fares[sorted_indices]
    mean_demands = mean_demands[sorted_indices]
    probabilities = probabilities[sorted_indices]
    
    # Initialize protection levels
    protection_levels = np.zeros(n)
    
    # EMSR-b algorithm
    for i in range(1, n):
        # Calculate weighted average fare for classes 1 to i
        weighted_fare = sum(fares[j] * mean_demands[j] for j in range(i)) / sum(mean_demands[j] for j in range(i))
        
        # Calculate protection level for fare classes 1 to i against class i+1
        # This simplified version uses the provided probabilities
        total_demand = sum(mean_demands[j] for j in range(i))
        adjusted_prob = sum(probabilities[j] * mean_demands[j] for j in range(i)) / total_demand if total_demand > 0 else 0
        
        # Protection level calculation (simple approach based on probability thresholds)
        if fares[i] > 0:
            threshold = 1 - (fares[i] / weighted_fare) if weighted_fare > 0 else 0
            protection_level = total_demand * min(max(adjusted_prob - threshold, 0), 1)
        else:
            protection_level = 0
            
        protection_levels[i-1] = protection_level
    
    # Calculate booking limits
    booking_limits = np.zeros(n)
    remaining = capacity
    for i in range(n):
        if i == 0:
            booking_limits[i] = min(remaining, mean_demands[i])
        else:
            booking_limits[i] = min(remaining, max(0, mean_demands[i] - protection_levels[i-1]))
        remaining -= booking_limits[i]
    
    # Calculate expected revenue
    expected_revenue = sum(fares[i] * booking_limits[i] * probabilities[i] for i in range(n))
    
    return protection_levels, booking_limits, expected_revenue, sorted_indices

def davn_emsr_b_integration(davn_matrix, product_to_legs, fare, demand, capacity):
    """
    Integrate DAVN output with EMSR-b calculation for each leg.
    
    Parameters:
    davn_matrix -- the DAVN bid price matrix from DAVN calculation
    product_to_legs -- mapping of products to legs
    fare -- original fares for each product
    demand -- original demand for each product
    capacity -- capacity for each leg
    
    Returns:
    results -- dictionary with EMSR-b results for each leg
    """
    NUMBER_OF_LEGS = capacity.shape[0]
    results = {}
    
    for leg in range(NUMBER_OF_LEGS):
        # Extract adjusted fares for this leg
        leg_fares, product_ids = extract_leg_fare_classes(davn_matrix, leg, product_to_legs, fare)
        
        if len(leg_fares) == 0:
            results[leg] = {
                "leg": leg,
                "message": "No valid fare classes for this leg",
                "protection_levels": [],
                "booking_limits": [],
                "expected_revenue": 0,
                "fares": [],
                "product_ids": []
            }
            continue
        
        # Get mean demands for these products
        leg_demands = np.array([demand[p] for p in product_ids])
        
        # Hardcoded probabilities for each fare class (adjust as needed)
        # Higher probabilities for higher fare classes (business logic)
        probabilities = np.linspace(0.95, 0.75, len(leg_fares))
        
        # Calculate EMSR-b
        protection_levels, booking_limits, expected_revenue, sorted_indices = calculate_emsr_b(
            leg_fares, 
            leg_demands, 
            probabilities,
            capacity[leg]
        )
        
        # Store results
        results[leg] = {
            "leg": leg,
            "protection_levels": protection_levels,
            "booking_limits": booking_limits, 
            "expected_revenue": expected_revenue,
            "fares": leg_fares[sorted_indices],
            "product_ids": product_ids[sorted_indices],
            "demands": leg_demands[sorted_indices],
            "probabilities": probabilities[sorted_indices]
        }
    
    return results

# Export the functions explicitly
__all__ = ['calculate_emsr_b', 'davn_emsr_b_integration']
