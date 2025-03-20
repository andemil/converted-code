import numpy as np
import matplotlib.pyplot as plt
from davn_utils import products_on_leg_finder, davn_generator, extract_leg_fare_classes
from davn_optimization import optimize_with_scipy

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

def run_davn_emsr_b_example():
    """Run a complete example of DAVN + EMSR-b integration"""
    # Define problem parameters (same as in davn_book.py)
    NUMBER_OF_PRODUCTS = 24
    NUMBER_OF_LEGS = 6

    # Define fares for each product
    fare = np.array([
        350, 375, 400, 430, 450, 500, 600, 610, 620, 630, 640, 650,
        500, 525, 550, 585, 600, 650, 750, 760, 770, 780, 790, 800
    ], dtype=float)

    # Define demand for each product
    demand = np.array([
        58.8, 67.2, 50.4, 58.8, 67.2, 50.4, 84, 100.8, 84, 75.6, 84, 58.8,
        14.7, 16.8, 12.6, 14.7, 16.8, 12.6, 21, 25.2, 21, 18.9, 21, 14.7
    ], dtype=float)

    # Define capacity for each leg
    capacity = np.array([100, 100, 100, 100, 100, 100], dtype=float)

    # Define product-to-legs mapping
    product_to_legs = np.array([
        [0, -1], [1, -1], [2, -1], [3, -1], [4, -1], [5, -1],
        [1, 2],  [0, 3],  [1, 4],  [0, 5],  [3, 4],  [2, 5],
        [0, -1], [1, -1], [2, -1], [3, -1], [4, -1], [5, -1],
        [1, 2],  [0, 3],  [1, 4],  [0, 5],  [3, 4],  [2, 5]
    ])

    # Run optimization using SciPy (replacing PuLP)
    results = optimize_with_scipy(fare, demand, capacity, product_to_legs)
    shadow_prices = results['shadow_prices']
    
    # Calculate DAVN matrix
    davn = davn_generator(shadow_prices, fare, product_to_legs, NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS)
    
    # Run EMSR-b for each leg using DAVN output
    results = davn_emsr_b_integration(davn, product_to_legs, fare, demand, capacity)
    
    # Display results
    print("\n=== DAVN and EMSR-b Integration Results ===\n")
    total_expected_revenue = 0
    
    for leg, result in results.items():
        print(f"\nLeg {leg}:")
        
        if "message" in result:
            print(result["message"])
            continue
            
        print(f"  Products: {result['product_ids']}")
        print(f"  Adjusted fares: {result['fares']}")
        print(f"  Demand: {result['demands']}")
        print(f"  Probabilities: {result['probabilities']}")
        print(f"  Protection levels: {result['protection_levels']}")
        print(f"  Booking limits: {result['booking_limits']}")
        print(f"  Expected revenue: ${result['expected_revenue']:.2f}")
        
        total_expected_revenue += result['expected_revenue']
    
    print(f"\nTotal expected revenue from all legs: ${total_expected_revenue:.2f}")
    
    # Visualize booking limits for each leg
    plot_results(results)
    
    return results

def plot_results(results):
    """Plot the booking limits and protection levels for each leg"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for leg, result in results.items():
        if "message" in result:
            axes[leg].text(0.5, 0.5, result["message"], 
                          ha='center', va='center', fontsize=12)
            axes[leg].set_title(f"Leg {leg}")
            continue
            
        fares = result["fares"]
        booking_limits = result["booking_limits"]
        protection_levels = result["protection_levels"]
        
        # Create labels for fare classes
        labels = [f"${fare:.0f}" for fare in fares]
        
        # Plot booking limits
        axes[leg].bar(labels, booking_limits, color='skyblue', alpha=0.7)
        axes[leg].set_title(f"Leg {leg} - Booking Limits")
        axes[leg].set_xlabel("Fare Class")
        axes[leg].set_ylabel("Seats")
        
        # Add values on top of bars
        for i, v in enumerate(booking_limits):
            axes[leg].text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/Users/prajitadhikari/Downloads/Optimization for all/converted code/davn_emsr_results.png')
    plt.show()

if __name__ == "__main__":
    run_davn_emsr_b_example()
