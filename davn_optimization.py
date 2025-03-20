import numpy as np
from scipy.optimize import linprog
from davn_utils import products_on_leg_finder, davn_generator

def optimize_with_scipy(fare, demand, capacity, product_to_legs):
    """Solve the DAVN optimization problem using SciPy's linprog"""
    NUMBER_OF_PRODUCTS = len(fare)
    NUMBER_OF_LEGS = len(capacity)
    
    # Objective function: linprog in Python minimizes, so we use -fare
    f = -fare

    # Build constraint matrix efficiently
    # First part: demand constraints (identity matrix)
    # Second part: capacity constraints
    A = np.vstack([
        np.eye(NUMBER_OF_PRODUCTS),  # Demand constraints (Z_i <= demand_i)
        np.zeros((NUMBER_OF_LEGS, NUMBER_OF_PRODUCTS))  # Capacity constraints placeholder
    ])

    # Fill in the capacity constraints
    for leg in range(NUMBER_OF_LEGS):
        prod_list = products_on_leg_finder(leg, product_to_legs)
        for j in prod_list:
            A[NUMBER_OF_PRODUCTS + leg, j] = 1

    # Right-hand side: demand followed by capacity
    b = np.concatenate([demand, capacity])

    # Lower bound = 0 for all variables
    bounds = [(0, None)] * NUMBER_OF_PRODUCTS

    # Solve LP using HiGHS solver
    options = {'disp': False}
    res = linprog(c=f, A_ub=A, b_ub=b, bounds=bounds, method='highs', options=options)

    x = res.x
    fval = res.fun
    exitflag = res.success

    # Get dual prices from the reduced costs
    try:
        # For newer versions of scipy with HiGHS
        dual_prices = res.ineqlin.marginals
    except AttributeError:
        try:
            # Alternative way to access duals
            dual_prices = res.shadow_prices
        except AttributeError:
            print("Warning: Using older SciPy version. Switching to revised simplex method.")
            # Fall back to revised simplex method
            res = linprog(c=f, A_ub=A, b_ub=b, bounds=bounds, method='revised simplex')
            dual_prices = res.slack

    if dual_prices is None:
        raise Exception("Could not obtain dual prices with any method")

    # Extract capacity shadow prices for the last NUMBER_OF_LEGS constraints:
    shadow_prices = dual_prices[NUMBER_OF_PRODUCTS:NUMBER_OF_PRODUCTS + NUMBER_OF_LEGS]
    
    # Calculate DAVN matrix
    davn_matrix = davn_generator(shadow_prices, fare, product_to_legs, NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS)
    
    return {
        'optimal_solution': x,
        'optimal_value': -fval,
        'shadow_prices': shadow_prices,
        'davn_matrix': davn_matrix
    }

# Removed the optimize_with_pulp function since it's not used
