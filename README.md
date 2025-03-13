# Optimization Comparison: SciPy vs PuLP

This directory contains two implementations of the DAVN (Displacement Adjusted Virtual Nesting) optimization model:
- `davn_book.py`: using SciPy's `linprog`
- `davn_book_pulp.py`: using PuLP

## Advantages of PuLP over SciPy.optimize

1. **Readability**: PuLP provides a more intuitive syntax for formulating optimization problems
2. **Flexibility**: PuLP can easily handle various types of constraints and objectives
3. **Solver Options**: PuLP supports multiple solvers (CBC, GLPK, CPLEX, Gurobi, etc.)
4. **Integer Programming**: PuLP natively supports integer and binary variables
5. **Model Manipulation**: Easier to modify constraints and objectives dynamically

## When to Use Each Library

**Use SciPy.optimize when:**
- Solving simple linear programming problems
- You need tight integration with other SciPy functions
- Minimizing dependencies in your project

**Use PuLP when:**
- Working with complex or large-scale optimization models
- You need integer variables or other advanced features
- Readability and maintainability are important
- You might need to switch between different solvers

## Implementation Notes

Both implementations solve the same linear programming problem, but the PuLP version is more readable and easier to extend if the problem requirements change in the future.

## DAVN and EMSR-b Integration

The repository now includes a new module `davn_emsr_b.py` that integrates DAVN with EMSR-b revenue management:

1. **DAVN (Displacement Adjusted Virtual Nesting)** calculates bid prices for each product-leg combination
2. **EMSR-b (Expected Marginal Seat Revenue with Bucketing)** uses these bid prices to determine optimal booking limits

### How It Works

1. First, the DAVN algorithm solves the revenue optimization problem using PuLP
2. The shadow prices from the solution are used to generate the DAVN bid price matrix
3. For each leg, relevant fare classes are extracted from the DAVN matrix
4. Hardcoded probabilities are assigned to each fare class
5. The EMSR-b algorithm calculates protection levels and booking limits
6. Results are visualized showing booking limits for each fare class on each leg

### Running the Integration

To run the integration:
```python
python davn_emsr_b.py
```

This will execute the full pipeline and generate visualizations of the booking limits.
