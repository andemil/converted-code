import numpy as np
import pulp

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

# Create a PuLP maximization problem
prob = pulp.LpProblem("Revenue_Maximization", pulp.LpMaximize)

# Create decision variables
Z = pulp.LpVariable.dicts("Z", range(NUMBER_OF_PRODUCTS), lowBound=0)

# Add objective function: maximize revenue
prob += pulp.lpSum([fare[i] * Z[i] for i in range(NUMBER_OF_PRODUCTS)])

# Add demand constraints
for i in range(NUMBER_OF_PRODUCTS):
    prob += Z[i] <= demand[i], f"Demand_Constraint_{i}"

# Add capacity constraints
for leg in range(NUMBER_OF_LEGS):
    # Get products using this leg
    prod_list = products_on_leg_finder(leg, product_to_legs)
    if prod_list:
        prob += pulp.lpSum([Z[j] for j in prod_list]) <= capacity[leg], f"Capacity_Constraint_{leg}"

# Solve the problem
prob.solve(pulp.PULP_CBC_CMD(msg=False))

# Extract results
x = np.array([Z[i].value() for i in range(NUMBER_OF_PRODUCTS)])
optimal_value = pulp.value(prob.objective)

# Get shadow prices (dual values) for the capacity constraints
shadow_prices = np.zeros(NUMBER_OF_LEGS)
for leg in range(NUMBER_OF_LEGS):
    constraint_name = f"Capacity_Constraint_{leg}"
    if constraint_name in prob.constraints:
        shadow_prices[leg] = prob.constraints[constraint_name].pi

print("Shadow prices (capacity constraints):", shadow_prices)
print("\nOptimal solution (Z0..Z23):")
print(x)
print("\nUpper bound on profit:")
print(optimal_value)

# Calculate DAVN matrix
davn = davn_generator(shadow_prices, fare, product_to_legs, NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS)
print("\nDAVN matrix:\n", davn)
