import numpy as np
from scipy.optimize import linprog


NUMBER_OF_PRODUCTS = 24
NUMBER_OF_LEGS = 6

fare = np.array([
    350, 375, 400, 430, 450, 500, 600, 610, 620, 630, 640, 650,
    500, 525, 550, 585, 600, 650, 750, 760, 770, 780, 790, 800
], dtype=float)

# Objective function: linprog in Python minimizes, so we use -fare
f = -fare

# Inequality matrix (A_ub) and RHS (b_ub)
A = np.array([
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z0 <= 58.8
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z1 <= 67.2
    [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z2 <= 50.4
    [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z3 <= 58.8
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z4 <= 67.2
    [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z5 <= 50.4
    [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z6 <= 84
    [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z7 <= 100.8
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z8 <= 84
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z9 <= 75.6
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],   # Z10 <= 84
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],   # Z11 <= 58.8
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],   # Z12 <= 14.7
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],   # Z13 <= 16.8
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],   # Z14 <= 12.6
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],   # Z15 <= 14.7
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],   # Z16 <= 16.8
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],   # Z17 <= 12.6
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],   # Z18 <= 21
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],   # Z19 <= 25.2
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],   # Z20 <= 21
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],   # Z21 <= 18.9
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],   # Z22 <= 21
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],   # Z23 <= 14.7

    # Capacity constraints (last 6 rows):
    [1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0],
    [0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0],
    [0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1],
    [0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0],
    [0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1,0],
    [0,0,0,0,0,1,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,1]
], dtype=float)

b = np.array([
    58.8,67.2,50.4,58.8,67.2,50.4,84,100.8,84,75.6,84,58.8,
    14.7,16.8,12.6,14.7,16.8,12.6,21,25.2,21,18.9,21,14.7,
    100,100,100,100,100,100
], dtype=float)

# Lower bound = 0 for all variables
lb = [(0, None)] * NUMBER_OF_PRODUCTS

# Solve LP using HiGHS solver
options = {'disp': False}
res = linprog(c=f, A_ub=A, b_ub=b, bounds=lb, method='highs', options=options)

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
        res = linprog(c=f, A_ub=A, b_ub=b, bounds=lb, method='revised simplex')
        dual_prices = res.slack

if dual_prices is None:
    raise Exception("Could not obtain dual prices with any method")

number_of_constraints = len(dual_prices)

# Extract capacity shadow prices for the last 6 constraints:
shadow_prices = dual_prices[NUMBER_OF_PRODUCTS:NUMBER_OF_PRODUCTS + NUMBER_OF_LEGS]

print("Shadow prices (capacity constraints):", shadow_prices)
print("\nOptimal solution (Z0..Z23):")
print(x)

print("\nUpper bound on profit from LP:")
print(-fval)

# Now we call davn_generator (defined in a separate file) to compute the DAVN matrix
# Suppose we do:
from davn_generator import davn_generator
davn = davn_generator(shadow_prices, fare)
print("\nDAVN matrix:\n", davn)
