import numpy as np
from scipy.optimize import linprog

# TODO: Streamlit version to add the input and calculate the  output (optimized).

'''
Note: 
Two issues:
- The way it generates the constraints doesn't do it in the linear way. Change the input data, changing the constraints. 
- Output from davn, it has to capture the products. (Read the paper). One has to identify the products. Run the embsr code and get the cost then. Dav+ embsr combine and airline ticket pricing.
'''

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

print("Shadow prices (capacity constraints):", shadow_prices)
print("\nOptimal solution (Z0..Z23):")
print(x)

print("\nUpper bound on profit from LP:")
print(-fval)

# Calculate DAVN matrix directly instead of importing from another file
davn = davn_generator(shadow_prices, fare, product_to_legs, NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS)
print("\nDAVN matrix:\n", davn)
print("\n DAVN Shape:", davn.shape)
