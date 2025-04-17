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
    ## Optimizing Seat Allocation to Maximize Airline Revenue

    Airlines face a daily challenge: how to best manage limited seat capacity on their flights while
    maximizing revenue. The key lies in offering the right seats, at the right price, to the right customers
    — and knowing when to say no to low-fare bookings in hopes of selling those seats at a higher price
    later.

    This is the essence of airline revenue management. Airlines don't just sell one type of ticket — they
    offer multiple fare classes for the same route, ranging from economy saver fares to flexible business
    class tickets. Each fare class has different prices and conditions, and customer demand for these fares
    fluctuates over time.

    To make the most out of every flight, airlines use smart optimization techniques to decide:
    
    • How many seats should be available at each fare level?
                
    • When should a cheaper fare stop being offered?
                
    • How can the value of a seat be assessed when multiple itineraries overlap across a network
    of flights?

    ### From Single Flights to Complex Networks
    
    The EMSR-b heuristic, helps airlines set limits on how many tickets to sell at each price level. It uses
    historical demand data to strike a balance: selling seats early at a lower price versus waiting to sell
    them later at a higher price.
    
    However, in real-world airline operations, things are more complicated. Many customers book trips
    that involve multiple connecting flights, and a seat on one leg of a journey may be shared across
    many different itineraries.
    
    To handle this, our approach uses a method called Displacement Adjusted Virtual Nesting (DAVN).
    This method helps airlines make smarter decisions by considering the entire flight network. It
    estimates the true revenue value of each ticket, accounting for the opportunity cost of assigning a
    seat to one itinerary instead of another.
    
    Using a combination of:
    
    • Linear programming to optimize revenue across the network, and
                
    • Heuristic rules to manage booking decisions at the individual flight level.
    
    This integrated system enables airlines to manage bookings in a way that is both strategically
    optimal and operationally practical.

    ### The Big Picture
    
    This solution gives airlines a powerful tool to improve profitability without adding more flights or
    seats. By making informed booking decisions backed by optimization models, airlines can increase
    revenue, manage uncertainty in demand, and better utilize their limited capacity.
    
    For readers interested in the full mathematical model and formulation, please see [this PDF](https://drive.google.com/file/d/1ezOJL3COfIRSI4bEX_xcEzGp3rK6fjuO/view?usp=sharing).
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
          
        - **Product-to-Legs Mapping**: This matrix is critical for the optimization algorithm:
          - Each row represents a product (product 1 is row 0, product 2 is row 1, etc.)
          - The first column specifies the first leg used by the product
          - The second column specifies the second leg used by the product (if applicable)
          - Values 0-{num_legs-1} indicate the leg index (leg 1 is represented by 0, leg 2 by 1, etc.)
          - Value -1 means "no leg" (for products that use only one leg)
          
          For example, in the default data:
          - Products 1-6 each use only one leg (legs 1-6 respectively)
          - Product 7 uses legs 2 and 3
          - Product 8 uses legs 1 and 4
          - And so on...
          
          This mapping is essential for the DAVN algorithm to work correctly.
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
    # Enhanced warning about valid leg indices and importance of this matrix
    st.warning("""
    **Important**: This matrix defines which legs each product uses. Without correct mapping, the optimization will return incorrect results.
    
    - Each row represents a product
    - First column = first leg used by the product
    - Second column = second leg used by the product (or -1 if only one leg is used)
    - Use values 0-{} to specify leg indices (0=leg 1, 1=leg 2, etc.)
    - Use -1 to indicate no leg
    """.format(num_legs-1))
    
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
