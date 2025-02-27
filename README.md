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
