import numpy as np

def products_on_leg_finder(leg, product_to_legs):
    """
    Return a list of product indices that use a given leg.
    """
    products = []
    for i, legs in enumerate(product_to_legs):
        if leg in legs:
            products.append(i)
    return products

def davn_generator(shadow_prices, fare, product_to_legs, NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS):
    """
    Compute the bid (or adjusted) prices for each product-leg combination.
    """
    davn = -np.ones((NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS))
    for product in range(NUMBER_OF_PRODUCTS):
        leg_set = [leg for leg in product_to_legs[product] if leg != -1]
        sum_shadow = sum(shadow_prices[l] for l in leg_set)
        for l in leg_set:
            davn[product, l] = fare[product] - sum_shadow + shadow_prices[l]
    return davn

def extract_leg_fare_classes(davn_matrix, leg, product_to_legs, fare):
    """
    Extract adjusted fares for a specific leg from the DAVN matrix.
    Only returns positive bid prices (valid fare classes).
    """
    # Get all products that use this leg
    products = products_on_leg_finder(leg, product_to_legs)
    
    # Extract adjusted fares for these products
    adjusted_fares = []
    product_ids = []
    for p in products:
        # Only consider valid adjusted fares (positive values)
        if davn_matrix[p, leg] > 0:
            adjusted_fares.append(davn_matrix[p, leg])
            product_ids.append(p)
    
    return np.array(adjusted_fares), np.array(product_ids)

def get_default_data():
    """Return default data for the optimization problem"""
    # Define problem parameters
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

    return {
        'NUMBER_OF_PRODUCTS': NUMBER_OF_PRODUCTS,
        'NUMBER_OF_LEGS': NUMBER_OF_LEGS,
        'fare': fare,
        'demand': demand,
        'capacity': capacity,
        'product_to_legs': product_to_legs
    }
