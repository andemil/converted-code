import numpy as np
from scipy.optimize import linprog

def leg_finder(product, product_to_legs):
    """
    Return the list of legs used by a given product.
    (Filter out entries equal to -1)
    """
    return [leg for leg in product_to_legs[product] if leg != -1]

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
    davn = -np.ones((NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS))
    for product in range(NUMBER_OF_PRODUCTS):
        leg_set = leg_finder(product, product_to_legs)
        sum_shadow = sum(shadow_prices[l] for l in leg_set)
        for l in leg_set:
            davn[product, l] = fares[product] - sum_shadow + shadow_prices[l]
    return davn

def network_revenue_management(fares=None, demand=None, capacity=None, product_to_legs=None):
    """
    Solve the network revenue management linear programming problem.
    If parameters are not provided, default values will be used.
    
    Parameters:
    -----------
    fares : array-like
        Fares for each product
    demand : array-like
        Expected demand for each product
    capacity : array-like
        Capacity for each leg
    product_to_legs : array-like
        Mapping of products to legs
        
    Returns:
    --------
    x : array
        Optimal solution
    optimal_profit : float
        Maximum achievable profit
    shadow_prices : array
        Shadow prices for leg capacity constraints
    davn : array
        Bid prices (DAVN matrix)
    """
    # Default values if parameters are not provided
    if fares is None:
        fares = np.array([
            350, 375, 400, 430, 450, 500, 600, 610, 620, 630, 640, 650,
            500, 525, 550, 585, 600, 650, 750, 760, 770, 780, 790, 800
        ])
    
    if demand is None:
        demand = np.array([
            58.8, 67.2, 50.4, 58.8, 67.2, 50.4, 84, 100.8, 84, 75.6, 
            84, 58.8, 14.7, 16.8, 12.6, 14.7, 16.8, 12.6, 21, 25.2,
            21, 18.9, 21, 14.7
        ])
    
    if capacity is None:
        capacity = np.array([100, 100, 100, 100, 100, 100])
    
    if product_to_legs is None:
        product_to_legs = np.array([
            [0, -1], [1, -1], [2, -1], [3, -1], [4, -1], [5, -1],
            [1, 2], [0, 3], [1, 4], [0, 5], [3, 4], [2, 5],
            [0, -1], [1, -1], [2, -1], [3, -1], [4, -1], [5, -1],
            [1, 2], [0, 3], [1, 4], [0, 5], [3, 4], [2, 5]
        ])
    
    # Determine dimensions from input data
    NUMBER_OF_PRODUCTS = len(fares)
    NUMBER_OF_LEGS = len(capacity)
    
    # Verify input dimensions are consistent
    if len(demand) != NUMBER_OF_PRODUCTS:
        raise ValueError("Demand array must have same length as fares array")
    if len(product_to_legs) != NUMBER_OF_PRODUCTS:
        raise ValueError("Product-to-legs mapping must have same number of rows as products")
    
    # Set up the LP
    f = -fares  # Objective: maximize revenue (minimize negative revenue)
    
    # Build constraint matrix efficiently using np.vstack
    A = np.vstack([
        np.eye(NUMBER_OF_PRODUCTS),  # Demand constraints
        np.zeros((NUMBER_OF_LEGS, NUMBER_OF_PRODUCTS))  # Capacity constraints placeholder
    ])
    
    # Fill in the capacity constraints
    for leg in range(NUMBER_OF_LEGS):
        prod_list = products_on_leg_finder(leg, product_to_legs)
        for j in prod_list:
            A[NUMBER_OF_PRODUCTS + leg, j] = 1
    
    # Right-hand side vector
    b = np.concatenate((demand, capacity))
    
    # Solve the LP
    options = {'disp': False}
    res = linprog(c=f, A_ub=A, b_ub=b, bounds=[(0, None)]*NUMBER_OF_PRODUCTS, 
                 method='highs', options=options)
    
    if not res.success:
        raise RuntimeError("LP solver did not converge: " + res.message)
    
    x = res.x
    optimal_profit = -res.fun
    
    # Extract dual prices
    try:
        shadow_all = res.ineqlin.marginals
    except AttributeError:
        try:
            shadow_all = res.shadow_prices
        except AttributeError:
            print("Warning: Using older SciPy version. Shadow prices may not be accurate.")
            shadow_all = np.zeros(NUMBER_OF_PRODUCTS + NUMBER_OF_LEGS)
    
    shadow_prices = shadow_all[NUMBER_OF_PRODUCTS : NUMBER_OF_PRODUCTS + NUMBER_OF_LEGS]
    
    # Print results
    print("Optimal solution (x for each product):")
    print(x)
    print("\nUpper bound on profit from LP:")
    print(optimal_profit)
    print("\nShadow prices for each leg (capacity constraints):")
    print(shadow_prices)
    
    # Compute DAVN matrix
    davn = davn_generator(shadow_prices, fares, product_to_legs, NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS)
    print("\nBid prices (davn matrix):")
    print(davn)
    
    return x, optimal_profit, shadow_prices, davn
