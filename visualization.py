import numpy as np
import matplotlib.pyplot as plt
import io
import base64

def plot_results(results):
    """Plot the booking limits and protection levels for each leg"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for leg, result in results.items():
        if "message" in result:
            axes[leg].text(0.5, 0.5, result["message"], 
                          ha='center', va='center', fontsize=12)
            axes[leg].set_title(f"Leg {leg}")
            continue
            
        fares = result["fares"]
        booking_limits = result["booking_limits"]
        protection_levels = result["protection_levels"]
        
        # Create labels for fare classes
        labels = [f"${fare:.0f}" for fare in fares]
        
        # Plot booking limits
        axes[leg].bar(labels, booking_limits, color='skyblue', alpha=0.7)
        axes[leg].set_title(f"Leg {leg} - Booking Limits")
        axes[leg].set_xlabel("Fare Class")
        axes[leg].set_ylabel("Seats")
        
        # Add values on top of bars
        for i, v in enumerate(booking_limits):
            axes[leg].text(i, v + 0.5, f"{v:.1f}", ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Convert plot to a base64 string for embedding in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return plot_data

def plot_davn_matrix(davn_matrix, product_to_legs):
    """Create a heatmap visualization of the DAVN matrix"""
    # Filter only valid entries (where product uses a leg)
    valid_mask = np.zeros_like(davn_matrix, dtype=bool)
    for product in range(davn_matrix.shape[0]):
        for leg in range(davn_matrix.shape[1]):
            leg_set = [l for l in product_to_legs[product] if l != -1]
            if leg in leg_set:
                valid_mask[product, leg] = True
    
    # Create a masked version of the matrix
    masked_davn = np.ma.array(davn_matrix, mask=~valid_mask)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    im = ax.imshow(masked_davn, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Bid Price ($)', rotation=-90, va="bottom")
    
    # Add labels and ticks
    ax.set_title('DAVN Bid Price Matrix')
    ax.set_xlabel('Leg')
    ax.set_ylabel('Product')
    
    # Set tick labels
    ax.set_xticks(np.arange(davn_matrix.shape[1]))
    ax.set_yticks(np.arange(davn_matrix.shape[0]))
    ax.set_xticklabels([f'Leg {i}' for i in range(davn_matrix.shape[1])])
    ax.set_yticklabels([f'Product {i}' for i in range(davn_matrix.shape[0])])
    
    # Rotate tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for product in range(davn_matrix.shape[0]):
        for leg in range(davn_matrix.shape[1]):
            if valid_mask[product, leg]:
                text = ax.text(leg, product, f"{davn_matrix[product, leg]:.0f}",
                              ha="center", va="center", color="w" if davn_matrix[product, leg] > 0 else "black")
    
    fig.tight_layout()
    
    # Convert plot to a base64 string for embedding in Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return plot_data
