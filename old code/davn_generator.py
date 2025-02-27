import numpy as np
from leg_finder import leg_finder
from number_leg_finder import number_leg_finder

NUMBER_OF_PRODUCTS = 24
NUMBER_OF_LEGS = 6

def davn_generator(shadow_prices, fare):
    """
    Replicates the logic of the MATLAB davn_generator function.
    shadow_prices: array of dual prices for each leg (length = NUMBER_OF_LEGS)
    fare: array of size NUMBER_OF_PRODUCTS
    Returns: a (NUMBER_OF_PRODUCTS x NUMBER_OF_LEGS) matrix of DAVN values
    """

    # Initialize all to -1
    davn = -1 * np.ones((NUMBER_OF_PRODUCTS, NUMBER_OF_LEGS), dtype=float)

    # Compute bid prices for valid product-leg combos
    for product in range(1, NUMBER_OF_PRODUCTS+1):
        leg_set = leg_finder(product)
        number_of_legs_used = number_leg_finder(leg_set)

        # Sum up the dual prices for the legs used
        sum_val = 0.0
        for i in range(number_of_legs_used):
            # leg indices in MATLAB are 1-based. If leg_set(i) = 2, that is leg #2 in MATLAB,
            # which in Python we might treat as index 1. So subtract 1 as needed.
            leg_index = leg_set[i]  # e.g. might be 2 or 3
            if leg_index > 0:       # i.e. != -1
                sum_val += shadow_prices[leg_index - 1]

        # Now fill in the DAVN values
        for i in range(number_of_legs_used):
            leg_index = leg_set[i]
            if leg_index > 0:
                # fare(product-1): because product is 1-based
                davn[product-1, leg_index-1] = (
                    fare[product-1] - sum_val + shadow_prices[leg_index-1]
                )

    return davn
