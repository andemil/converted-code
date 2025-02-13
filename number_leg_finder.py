def number_leg_finder(leg_set):
    """
    Computes how many valid legs appear in leg_set (ignoring any -1).
    leg_set is a 2-element list, possibly [2, -1], etc.
    """
    count = 0
    for val in leg_set:
        if val > 0:
            count += 1
    return count
