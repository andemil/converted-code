import streamlit as st
import numpy as np
import pandas as pd
import base64
from io import BytesIO

# Import custom modules - removing PuLP dependencies
from davn_utils import get_default_data, extract_leg_fare_classes
from davn_optimization import optimize_with_scipy
from davn_emsr_b import davn_emsr_b_integration
from visualization import plot_results, plot_davn_matrix

###########################################
# APPLICATION CONFIGURATION
###########################################

# Set page title and configuration
st.set_page_config(
    page_title="DAVN & EMSR-b Optimizer",
    page_icon="✈️",
    layout="wide"
)

# Function to run the full optimization pipeline
def run_optimization(data):
    # Extract data
    fare = data['fare']
    demand = data['demand']
    capacity = data['capacity']
    product_to_legs = data['product_to_legs']
    
    # Run optimization using SciPy only
    results = optimize_with_scipy(fare, demand, capacity, product_to_legs)
    
    # Extract DAVN matrix
    davn_matrix = results['davn_matrix']
    
    # Run EMSR-b integration
    emsr_results = davn_emsr_b_integration(davn_matrix, product_to_legs, fare, demand, capacity)
    
    # Calculate total expected revenue
    total_revenue = sum(result['expected_revenue'] for leg, result in emsr_results.items() if 'expected_revenue' in result)
    
    return {
        'optimization_results': results,
        'emsr_results': emsr_results,
        'total_revenue': total_revenue
    }

# Function to display optimization results
def display_results(results, data):
    # Extract results
    opt_results = results['optimization_results']
    emsr_results = results['emsr_results']
    total_revenue = results['total_revenue']
    
    # Display optimization summary
    st.header("Optimization Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Optimal Value (LP)", f"${opt_results['optimal_value']:.2f}")
        
    with col2:
        st.metric("Expected Revenue (EMSR-b)", f"${total_revenue:.2f}")
    
    # Display shadow prices
    st.subheader("Shadow Prices (Capacity Constraints)")
    shadow_prices_df = pd.DataFrame({
        'Leg': range(len(opt_results['shadow_prices'])),
        'Shadow Price': opt_results['shadow_prices']
    })
    st.dataframe(shadow_prices_df)
    
    # DAVN Matrix visualization
    st.subheader("DAVN Bid Price Matrix")
    davn_plot = plot_davn_matrix(opt_results['davn_matrix'], data['product_to_legs'])
    st.image(f"data:image/png;base64,{davn_plot}")  # removed use_container_width=True
    
    # EMSR-b Results
    st.header("EMSR-b Results by Leg")
    
    booking_limits_plot = plot_results(emsr_results)
    st.image(f"data:image/png;base64,{booking_limits_plot}")  # removed use_container_width=True
    
    # Detailed EMSR-b results in expandable sections
    for leg, result in emsr_results.items():
        with st.expander(f"Leg {leg+1} Details"):
            if "message" in result:
                st.write(result["message"])
                continue
                
            # Create a DataFrame for the leg results
            leg_df = pd.DataFrame({
                'Product ID': result['product_ids'],
                'Fare': result['fares'],
                'Demand': result['demands'],
                'Probability': result['probabilities'],
                'Protection Level': result['protection_levels'] if len(result['protection_levels']) == len(result['fares']) else np.append(result['protection_levels'], 0),
                'Booking Limit': result['booking_limits']
            })
            
            st.dataframe(leg_df)
            st.metric("Expected Revenue", f"${result['expected_revenue']:.2f}")

###########################################
# DOCUMENTATION
###########################################

def show_methodology_information():
    st.markdown("""
    ## Introduction to Airline Revenue Management

    Airline revenue management revolves around the challenge of determining which customer bookings to accept and which to reject, aiming to maximize overall revenue. Airlines offer multiple fare classes for the same origin-destination itinerary, each associated with different conditions and privileges. As tickets are sold over time, airlines must strategically adjust fare availability, closing lower fare classes and opening higher ones to optimize profitability.

    The key decision in revenue management is how many seats to allocate to each fare class to strike a balance between high occupancy and maximizing revenue. Allocating too many seats to lower fares may lead to full flights but lower profits, while reserving too many seats for higher fares risks unsold inventory, reducing potential revenue.

    ## Definitions of Key Concepts

    - **Fare Class** ($i$): A category of airline tickets distinguished by price and associated conditions.
    - **Fare Price** ($f_i$): The price of fare class $i$.
    - **Protection Level** ($Q_i$): The number of seats reserved for fare classes higher than and including $i$.
    - **Booking Limit** ($B_i$): The maximum number of seats that can be sold at a specific fare class $i$.
    - **Demand** ($D_i$): The number of bookings for fare class $i$.
    - **Leg** ($\\ell$): A single nonstop flight segment operated between two airports.
    - **Product**: An itinerary (which may contain one or more legs) combined with a fare class.

    ## EMSR-b Heuristic: Booking Limits for a Single Leg

    Littlewood's rule provides the foundation for airline revenue management by addressing the trade-off between selling a seat at a lower fare now versus reserving it for a potentially higher-paying customer in the future. Mathematically, in a two-fare class system, Littlewood's rule states that a seat should be sold at fare $f_1$ if:

    $$f_1 \\geq f_2 \\cdot \\text{Pr}[D_2 > Q_2]$$

    where $\\text{Pr}[D_2 > Q_2]$ represents the probability that demand $D_2$ for the higher fare $f_2$ exceeds the protection level $Q_2$.

    The EMSR-b heuristic generalizes Littlewood's rule to multiple fare classes on a single leg. The process consists of these steps:

    1. **Sort Fare Classes**: Arrange all fare classes in ascending order by fare price.
    2. **Compute Aggregated Demand and Fares**: For each fare class $i$, define the aggregated demand and fares for all fare classes above and including $i$.
    3. **Compute Protection Levels**: Determine the protection levels by solving Littlewood's equation.
    4. **Compute Booking Limits**: The booking limit for each fare class $i$ is given by $B_i = C - Q_{i+1}$, where $C$ is the aircraft capacity.

    ## DAVN Heuristic: Extending EMSR-b to Networks

    While EMSR-b optimizes seat allocations for a single flight leg, the Displacement Adjusted Virtual Nesting (DAVN) heuristic extends this concept to multi-leg itineraries.

    ### Linear Program Formulation
    DAVN solves the following linear program:

    $$\\max \\sum_{j=1}^{n} f_j x_j$$

    subject to:
    $$0 \\leq x_j \\leq E[D_j], \\quad \\forall j = 1, 2, ..., n$$
    $$\\sum_{j \\in A_\\ell} x_j \\leq C_\\ell, \\quad \\forall \\ell = 1, 2, ..., L$$

    where $f_j$ is the fare price of product $j$, $E[D_j]$ is the expected demand for product $j$, $C_\\ell$ is the capacity of leg $\\ell$, and $A_\\ell$ is the set of products using leg $\\ell$.

    ### Computing Displacement Adjusted Revenue (DARE)
    The displacement adjusted revenue for product $j$ on leg $\\ell$ is computed as:

    $$DARE_j^\\ell = f_j - \\sum_{i \\neq \\ell} \\lambda_i$$

    where $\\lambda_\\ell$ is the dual price (shadow price) of the capacity constraint for leg $\\ell$. These DARE values are then used as fares to apply EMSR-b on each leg separately, in order to determine the booking limits for each leg.

    ### Integrated Approach
    This application combines both methods:
    1. DAVN optimization to calculate network-level bid prices using linear programming
    2. EMSR-b to determine leg-level booking controls based on displacement-adjusted revenues

    This approach provides a comprehensive revenue management solution for airline capacity allocation, taking into account network effects while maintaining practical booking controls at the leg level.
    """)

###########################################
# DATA MANAGEMENT FUNCTIONS
###########################################

def edit_data():
    """Function to display and edit the default dataset"""
    data = get_default_data()
    
    st.subheader("Data Editor")
    
    with st.expander("Data Explanation", expanded=True):
        st.markdown("""
        ### Understanding the Data Structure
        
        - **Fares**: Each row represents a product (combination of itinerary and fare class).
          The fare is the price of that product.
          
        - **Demand**: Each row represents the expected demand for a product.
          
        - **Capacity**: Each row represents the seat capacity for a leg (flight segment).
          
        - **Product-to-Legs Mapping**: This matrix shows which legs are used by each product.
          - Each row corresponds to a product
          - Each product can use up to 2 legs
          - Values 0-5 indicate which leg is used (leg index)
          - Value -1 means "no leg" (for products that use only one leg)
          - Column 0 is the first leg of the product, Column 1 is the second leg (if applicable)
        """)
    
    # Add controls to change the number of products and legs
    # st.subheader("Configuration Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        current_num_products = len(data['fare'])
        num_products = st.number_input("Number of Products", min_value=1, max_value=100, value=current_num_products, step=1)
    
    with col2:
        current_num_legs = len(data['capacity'])
        num_legs = st.number_input("Number of Legs", min_value=1, max_value=50, value=current_num_legs, step=1)
    
    # Resize data structures if the number of products or legs changes
    if num_products != current_num_products or num_legs != current_num_legs:
        # Resize fare and demand arrays
        if num_products > current_num_products:
            # Add new products with default values
            data['fare'] = np.append(data['fare'], np.ones(num_products - current_num_products) * 500)  # Default fare: 500
            data['demand'] = np.append(data['demand'], np.ones(num_products - current_num_products) * 50)  # Default demand: 50
        else:
            # Truncate to fewer products
            data['fare'] = data['fare'][:num_products]
            data['demand'] = data['demand'][:num_products]
        
        # Resize capacity array
        if num_legs > current_num_legs:
            # Add new legs with default capacity
            data['capacity'] = np.append(data['capacity'], np.ones(num_legs - current_num_legs) * 100)  # Default capacity: 100
        else:
            # Truncate to fewer legs
            data['capacity'] = data['capacity'][:num_legs]
        
        # Resize product_to_legs mapping
        new_mapping = np.ones((num_products, 2), dtype=int) * -1  # Default: no leg (-1)
        
        # Copy existing values where possible
        min_products = min(num_products, current_num_products)
        for i in range(min_products):
            for j in range(min(2, data['product_to_legs'].shape[1])):
                # Only copy valid leg indices that still exist
                leg_idx = data['product_to_legs'][i, j]
                if leg_idx >= 0 and leg_idx < num_legs:
                    new_mapping[i, j] = leg_idx
                else:
                    new_mapping[i, j] = -1  # Invalid leg becomes "no leg"
        
        data['product_to_legs'] = new_mapping
    
    # Continue with data editing as before
    col1, col2 = st.columns(2)
    
    with col1:
        # Create editable dataframes with row labels
        st.subheader("Fares (by Product)")
        fares_df = pd.DataFrame({
            "Product": [f"Product {i+1}" for i in range(len(data['fare']))],
            "Fare": data['fare']
        }).set_index("Product")
        fares_edited = st.data_editor(fares_df)
        data['fare'] = fares_edited["Fare"].values
        
        st.subheader("Capacity (by Leg)")
        capacity_df = pd.DataFrame({
            "Leg": [f"Leg {i+1}" for i in range(len(data['capacity']))],
            "Capacity": data['capacity']
        }).set_index("Leg")
        capacity_edited = st.data_editor(capacity_df)
        data['capacity'] = capacity_edited["Capacity"].values
    
    with col2:
        st.subheader("Demand (by Product)")
        demand_df = pd.DataFrame({
            "Product": [f"Product {i+1}" for i in range(len(data['demand']))],
            "Demand": data['demand']
        }).set_index("Product")
        demand_edited = st.data_editor(demand_df)
        data['demand'] = demand_edited["Demand"].values
    
    st.subheader("Product-to-Legs Mapping")
    # Display a warning about valid leg indices
    st.info(f"For the mapping below, use values 0-{num_legs-1} to specify which leg is used, or -1 for 'no leg'.")
    # Create a dataframe with row and column labels for better understanding
    mapping_df = pd.DataFrame(
        data['product_to_legs'],
        index=[f"Product {i+1}" for i in range(data['product_to_legs'].shape[0])],
        columns=["First Leg", "Second Leg"]
    )
    mapping_edited = st.data_editor(mapping_df)
    
    # Validate leg indices to ensure they're within range
    valid_mapping = mapping_edited.values.copy()
    for i in range(valid_mapping.shape[0]):
        for j in range(valid_mapping.shape[1]):
            if valid_mapping[i, j] >= num_legs or valid_mapping[i, j] < -1:
                valid_mapping[i, j] = -1  # Reset invalid indices to -1
    
    data['product_to_legs'] = valid_mapping
    
    # Update the data structure with new dimensions
    data['NUMBER_OF_PRODUCTS'] = num_products
    data['NUMBER_OF_LEGS'] = num_legs
    
    return data

###########################################
# MAIN APPLICATION
###########################################

def main():
    # Application title and header
    st.title("✈️ Airline Revenue Management")
    st.write("### DAVN and EMSR-b Optimization Tool")

    # Documentation
    with st.expander("About the Optimization Methods"):
        show_methodology_information()
    
    
    # Display editable data directly (no option to switch to custom upload)
    data = edit_data()
    
    # Run optimization button
    if st.button("Run Optimization"):
        with st.spinner("Running optimization..."):
            # Run the optimization
            results = run_optimization(data)
            
            # Store results in session state for persistence between reruns
            st.session_state.results = results
            st.session_state.data = data
    
    # Display results if they exist in session state
    if 'results' in st.session_state and 'data' in st.session_state:
        display_results(st.session_state.results, st.session_state.data)
    else:
        # Show instructions if no results yet
        st.info("Click 'Run Optimization' to see the results.")

# Run the application
if __name__ == "__main__":
    main()
