import streamlit as st
import numpy as np
import pandas as pd
import base64
from io import BytesIO

# Import custom modules - modifying imports to address the error
from davn_utils import get_default_data, extract_leg_fare_classes
from davn_optimization import optimize_with_scipy, optimize_with_pulp

# Import directly from the davn_emsr_b.py file instead of emsr_b.py
from davn_emsr_b import davn_emsr_b_integration
from visualization import plot_results, plot_davn_matrix

# Set page title and configuration
st.set_page_config(
    page_title="DAVN & EMSR-b Optimizer",
    page_icon="✈️",
    layout="wide"
)

# Function to run the full optimization pipeline
def run_optimization(data, optimization_method='pulp'):
    # Extract data
    fare = data['fare']
    demand = data['demand']
    capacity = data['capacity']
    product_to_legs = data['product_to_legs']
    
    # Run optimization
    if optimization_method == 'pulp':
        results = optimize_with_pulp(fare, demand, capacity, product_to_legs)
    else:
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
    st.image(f"data:image/png;base64,{davn_plot}", use_column_width=True)
    
    # EMSR-b Results
    st.header("EMSR-b Results by Leg")
    
    booking_limits_plot = plot_results(emsr_results)
    st.image(f"data:image/png;base64,{booking_limits_plot}", use_column_width=True)
    
    # Detailed EMSR-b results in expandable sections
    for leg, result in emsr_results.items():
        with st.expander(f"Leg {leg} Details"):
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

# Function to handle file upload and data preparation
def process_uploaded_data(fare_file, demand_file, capacity_file, product_mapping_file):
    # Process fare data
    fare_content = fare_file.read()
    fare = np.loadtxt(BytesIO(fare_content))
    
    # Process demand data
    demand_content = demand_file.read()
    demand = np.loadtxt(BytesIO(demand_content))
    
    # Process capacity data
    capacity_content = capacity_file.read()
    capacity = np.loadtxt(BytesIO(capacity_content))
    
    # Process product mapping data
    mapping_content = product_mapping_file.read()
    product_to_legs = np.loadtxt(BytesIO(mapping_content), dtype=int)
    
    # Ensure dimensions are correct
    NUMBER_OF_PRODUCTS = len(fare)
    NUMBER_OF_LEGS = len(capacity)
    
    return {
        'NUMBER_OF_PRODUCTS': NUMBER_OF_PRODUCTS,
        'NUMBER_OF_LEGS': NUMBER_OF_LEGS,
        'fare': fare,
        'demand': demand,
        'capacity': capacity,
        'product_to_legs': product_to_legs
    }

# Function to generate downloadable CSV template
def get_csv_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Main Streamlit app
def main():
    st.title("✈️ Airline Revenue Management")
    st.write("### DAVN and EMSR-b Optimization Tool")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    optimization_method = st.sidebar.radio(
        "Choose Optimization Method",
        ("PuLP", "SciPy"),
        index=0
    )
    
    # Data input options
    data_source = st.sidebar.radio(
        "Data Source",
        ("Use Default Data", "Upload Custom Data"),
        index=0
    )
    
    # Load or prepare data
    if data_source == "Use Default Data":
        data = get_default_data()
        
        # Allow editing of default data
        if st.sidebar.checkbox("Edit Default Data"):
            with st.expander("Edit Default Data"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create editable dataframes
                    st.subheader("Fares")
                    fares_df = pd.DataFrame({"Fare": data['fare']})
                    fares_edited = st.data_editor(fares_df)
                    data['fare'] = fares_edited["Fare"].values
                    
                    st.subheader("Capacity")
                    capacity_df = pd.DataFrame({"Capacity": data['capacity']})
                    capacity_edited = st.data_editor(capacity_df)
                    data['capacity'] = capacity_edited["Capacity"].values
                
                with col2:
                    st.subheader("Demand")
                    demand_df = pd.DataFrame({"Demand": data['demand']})
                    demand_edited = st.data_editor(demand_df)
                    data['demand'] = demand_edited["Demand"].values
                    
                st.subheader("Product-to-Legs Mapping")
                mapping_df = pd.DataFrame(data['product_to_legs'])
                mapping_edited = st.data_editor(mapping_df)
                data['product_to_legs'] = mapping_edited.values
                
    else:
        # File upload section
        st.sidebar.subheader("Upload Data Files")
        
        # Allow downloading templates
        st.sidebar.markdown("### Download Templates")
        
        # Create template dataframes
        default_data = get_default_data()
        fares_template = pd.DataFrame({"Fare": default_data['fare']})
        demand_template = pd.DataFrame({"Demand": default_data['demand']})
        capacity_template = pd.DataFrame({"Capacity": default_data['capacity']})
        mapping_template = pd.DataFrame(default_data['product_to_legs'])
        
        # Add download links
        st.sidebar.markdown(get_csv_download_link(fares_template, "fares_template.csv", "Download Fares Template"), unsafe_allow_html=True)
        st.sidebar.markdown(get_csv_download_link(demand_template, "demand_template.csv", "Download Demand Template"), unsafe_allow_html=True)
        st.sidebar.markdown(get_csv_download_link(capacity_template, "capacity_template.csv", "Download Capacity Template"), unsafe_allow_html=True)
        st.sidebar.markdown(get_csv_download_link(mapping_template, "mapping_template.csv", "Download Mapping Template"), unsafe_allow_html=True)
        
        # File uploaders
        fare_file = st.sidebar.file_uploader("Upload Fares (CSV)", type=["csv", "txt"])
        demand_file = st.sidebar.file_uploader("Upload Demand (CSV)", type=["csv", "txt"])
        capacity_file = st.sidebar.file_uploader("Upload Capacity (CSV)", type=["csv", "txt"])
        product_mapping_file = st.sidebar.file_uploader("Upload Product-to-Legs Mapping (CSV)", type=["csv", "txt"])
        
        # Process uploaded files
        if fare_file and demand_file and capacity_file and product_mapping_file:
            data = process_uploaded_data(fare_file, demand_file, capacity_file, product_mapping_file)
        else:
            st.info("Please upload all required files or use default data.")
            return
    
    # Run optimization button
    if st.button("Run Optimization"):
        with st.spinner("Running optimization..."):
            # Run the optimization
            results = run_optimization(data, optimization_method.lower())
            
            # Store results in session state for persistence between reruns
            st.session_state.results = results
            st.session_state.data = data
            
            # No need to display results here, will be handled by the code below
    
    # Display results if they exist in session state
    # This handles both initial calculation and when the app reruns
    if 'results' in st.session_state and 'data' in st.session_state:
        display_results(st.session_state.results, st.session_state.data)
    else:
        # Show instructions if no results yet
        st.info("Click 'Run Optimization' to see the results.")
    
    # Add information section
    with st.expander("About the Optimization Methods"):
        st.markdown("""
        ### DAVN (Displacement Adjusted Virtual Nesting)
        DAVN is a network revenue management technique that calculates bid prices for each product-leg combination, 
        accounting for network effects and opportunity costs. It helps airlines decide which bookings to accept to maximize revenue.
        
        ### EMSR-b (Expected Marginal Seat Revenue with Bucketing)
        EMSR-b is a revenue management algorithm that determines optimal protection levels and booking limits for 
        different fare classes. It helps allocate capacity among fare classes to maximize expected revenue.
        
        ### Integrated Approach
        This application combines both methods:
        1. DAVN optimization to calculate network-level bid prices
        2. EMSR-b to determine leg-level booking controls
        
        This approach provides a comprehensive revenue management solution for airline capacity allocation.
        """)

if __name__ == "__main__":
    main()
