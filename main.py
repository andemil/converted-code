import numpy as np
from network_revenue_management import network_revenue_management
from emsr_b import emsr_b

def run_network_example():
    """Run the network revenue management example with default parameters"""
    print("=== Network Revenue Management LP (Default Parameters) ===")
    network_revenue_management()

def run_emsr_example():
    """Run the EMSR-b example with default parameters"""
    print("\n=== EMSR-b Computation (Default Parameters) ===")
    emsr_b()

def run_custom_network_example():
    """Example of running network revenue management with custom parameters"""
    print("\n=== Network Revenue Management LP (Custom Parameters) ===")
    # Example of custom parameters
    fares = np.array([400, 450, 500, 550])
    demand = np.array([60, 70, 50, 40])
    capacity = np.array([100, 100])
    product_to_legs = np.array([
        [0, -1],
        [1, -1],
        [0, 1],
        [0, 1]
    ])
    network_revenue_management(fares, demand, capacity, product_to_legs)

def run_custom_emsr_example():
    """Example of running EMSR-b with custom parameters"""
    print("\n=== EMSR-b Computation (Custom Parameters) ===")
    # Example of custom parameters  
    fares = np.array([100, 150, 200, 300])
    prob_class = np.array([0.4, 0.3, 0.2, 0.1])
    capacity = 150
    cancel_prob = 0.15
    emsr_b(fares, cancel_prob, capacity, arrival_rate=1.2, prob_class=prob_class)

if __name__ == "__main__":
    run_network_example()
    run_emsr_example()
    
    # Uncomment to run examples with custom parameters
    # run_custom_network_example()
    # run_custom_emsr_example()
