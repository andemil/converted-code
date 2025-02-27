import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson

# --- Helper functions for network revenue management ---

def leg_finder(product, product_to_legs):
    """
    Return the list of legs used by a given product.
    (Filter out entries equal to -1)
    """
    # product index is assumed to be 0-based; product_to_legs stores leg numbers (0-based)
    return [leg for leg in product_to_legs[product] if leg != -1]

def number_leg_finder(leg_set):
    """
    Return the number of legs in a leg_set.
    """
    return len(leg_set)

def products_on_leg_finder(leg, product_to_legs):
    """
    Return a list of product indices that use a given leg.
    """
    products = []
    for i, legs in enumerate(product_to_legs):
        if leg in legs:
            products.append(i)
    return products

def davn_generator(shadow_prices, fares, product_to_legs, NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS):
    """
    Compute the bid (or adjusted) prices for each product-leg combination.
    The output is a NUMBER_OF_PRODUCTS x NUMBER_OF_LEGS matrix.
    """
    # Initialize davn with -1
    davn = -np.ones((NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS))
    for product in range(NUMBER_OF_PRODUCTS):
        leg_set = leg_finder(product, product_to_legs)
        num_legs_used = number_leg_finder(leg_set)
        # Sum shadow prices over all legs used by the product
        sum_shadow = sum(shadow_prices[l] for l in leg_set)
        # For each leg used, compute bid price: fare - (sum of all shadow prices) + shadow price of that leg.
        for l in leg_set:
            davn[product, l] = fares[product] - sum_shadow + shadow_prices[l]
    return davn

# --- Main function for network revenue management LP ---

def network_revenue_management():
    # Global parameters
    NUMBER_OF_PRODUCTS = 24
    NUMBER_OF_LEGS = 6

    # Define fares (for each product)
    fares = np.array([
         350, 375, 400, 430, 450, 500, 600, 610, 620, 630, 640, 650,
         500, 525, 550, 585, 600, 650, 750, 760, 770, 780, 790, 800
    ])
    
    # Demand for each product (as a column vector)
    demand = np.array([
         58.8, 67.2, 50.4, 58.8, 67.2, 50.4, 84,   100.8, 84,   75.6, 
         84,   58.8, 14.7, 16.8, 12.6, 14.7, 16.8, 12.6, 21,   25.2,
         21,   18.9, 21,   14.7
    ])
    
    # Capacity on each leg
    capacity = np.array([100, 100, 100, 100, 100, 100])
    
    # Define the mapping of products to legs.
    # In the MATLAB code, leg numbers are 1-indexed.
    # Here, we convert them to 0-indexed.
    product_to_legs = np.array([
        [0, -1],
        [1, -1],
        [2, -1],
        [3, -1],
        [4, -1],
        [5, -1],
        [1, 2],
        [0, 3],
        [1, 4],
        [0, 5],
        [3, 4],
        [2, 5],
        [0, -1],
        [1, -1],
        [2, -1],
        [3, -1],
        [4, -1],
        [5, -1],
        [1, 2],
        [0, 3],
        [1, 4],
        [0, 5],
        [3, 4],
        [2, 5]
    ])
    
    # Set up the LP
    # Decision variables: x (number of units sold for each product)
    # We maximize revenue = sum(fares * x)
    # Since linprog minimizes, we set f = -fares.
    f = -fares

    # Number of LP constraints = (PRODUCT constraints + LEG capacity constraints)
    num_constraints = NUMBER_OF_PRODUCTS + NUMBER_OF_LEGS

    # Build matrix A for inequalities A*x <= b
    A = np.zeros((num_constraints, NUMBER_OF_PRODUCTS))
    
    # First NUMBER_OF_PRODUCTS rows: x <= demand (identity matrix)
    A[:NUMBER_OF_PRODUCTS, :] = np.eye(NUMBER_OF_PRODUCTS)
    
    # Next NUMBER_OF_LEGS rows: capacity constraints for each leg.
    # For each leg, sum over all products that use that leg must be <= capacity.
    for leg in range(NUMBER_OF_LEGS):
        prod_list = products_on_leg_finder(leg, product_to_legs)
        for j in prod_list:
            A[NUMBER_OF_PRODUCTS + leg, j] = 1

    # Right-hand side vector: first products' demands, then legs' capacities.
    b = np.concatenate((demand, capacity))
    
    # Lower bounds: x >= 0
    lb = np.zeros(NUMBER_OF_PRODUCTS)
    
    # Solve the linear program using SciPy's linprog (using the 'highs' method)
    res = linprog(c=f, A_ub=A, b_ub=b, bounds=[(0, None)]*NUMBER_OF_PRODUCTS, method='highs')
    if not res.success:
        raise RuntimeError("LP solver did not converge: " + res.message)
    
    x = res.x
    optimal_profit = -res.fun  # since we minimized -revenue
    
    # Extract dual variables (shadow prices) for inequality constraints.
    # With the 'highs' method, the dual values for A_ub constraints are available in res.ineqlin.marginals.
    # (Indices NUMBER_OF_PRODUCTS to end correspond to leg capacity constraints.)
    shadow_all = res.ineqlin.marginals
    shadow_prices = shadow_all[NUMBER_OF_PRODUCTS : NUMBER_OF_PRODUCTS + NUMBER_OF_LEGS]
    
    print("Optimal solution (x for each product):")
    print(x)
    print("\nUpper bound on profit from LP:")
    print(optimal_profit)
    print("\nShadow prices for each leg (capacity constraints):")
    print(shadow_prices)
    
    # Compute bid prices (davn) for each product-leg combination
    davn = davn_generator(shadow_prices, fares, product_to_legs, NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS)
    print("\nBid prices (davn matrix):")
    print(davn)
    
    return x, optimal_profit, shadow_prices, davn

# --- EMSR-b Implementation ---

def emsr_b():
    """
    Compute EMSR-b protection levels and booking limits for a set of fare classes.
    """
    # Fares sorted in increasing order
    fares = np.array([101, 127, 153, 179, 293, 419])
    cancel_prob = 0.20
    C = 100  # Aircraft seat capacity
    
    arrival_rate = 1.4   # per day (assumed Poisson process)
    time_horizon = 100   # days
    prob_class = np.array([0.3, 0.3, 0.13, 0.13, 0.09, 0.05])
    num_classes = len(fares)
    
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
        # Use the Poisson inverse (percent point function)
        protection_levels[i+1] = poisson.ppf(fraction, aggregate_demand[i+1])
    
    # Adjust capacity for cancellations
    C_adjusted = C / (1 - cancel_prob)
    
    # Compute booking limits for each fare class
    booking_limits = np.zeros(num_classes)
    for i in range(num_classes - 1):
        booking_limits[i] = max(C_adjusted - protection_levels[i+1], 0)
    booking_limits[num_classes - 1] = C_adjusted
    
    print("\nEMSR-b Results (Fare Class and Booking Limit):")
    for i in range(num_classes):
        print(f"Fare: {fares[i]:>4} | Booking Limit: {booking_limits[i]:.2f}")
    
    return fares, booking_limits

# --- Main Execution ---

if __name__ == "__main__":
    print("=== Network Revenue Management LP ===")
    network_revenue_management()
    
    print("\n=== EMSR-b Computation ===")
    emsr_b()
