def leg_finder(product):
    """
    Returns the array of leg numbers used by a given product.
    The product_to_legs matrix is ported from the MATLAB code:
    row i => product i
    Each row has up to two legs, or -1 to fill.
    """
    product_to_legs = [
        [1, -1],  # Product 1
        [2, -1],  # Product 2
        [3, -1],  # Product 3
        [4, -1],  # Product 4
        [5, -1],  # Product 5
        [6, -1],  # Product 6
        [2,  3],  # Product 7
        [1,  4],  # Product 8
        [2,  5],  # Product 9
        [1,  6],  # Product 10
        [4,  5],  # Product 11
        [3,  6],  # Product 12
        [1, -1],  # Product 13
        [2, -1],  # Product 14
        [3, -1],  # Product 15
        [4, -1],  # Product 16
        [5, -1],  # Product 17
        [6, -1],  # Product 18
        [2,  3],  # Product 19
        [1,  4],  # Product 20
        [2,  5],  # Product 21
        [1,  6],  # Product 22
        [4,  5],  # Product 23
        [3,  6]   # Product 24
    ]

    # product is 1-based in the original code, Python lists are 0-based
    idx = product - 1
    if idx < 0 or idx >= len(product_to_legs):
        raise ValueError("Invalid product index")

    return product_to_legs[idx]
