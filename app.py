import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set Streamlit page layout
st.set_page_config(layout="wide")

# Title of the web app
st.title("Pass Completion Heatmap Generator")

# File uploader widget for Excel files
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

# Proceed only if a file is uploaded
if uploaded_file:
    # Read the Excel file into a DataFrame
    df = pd.read_excel(uploaded_file, engine="openpyxl")

    # Filter only rows where the play type is 'Pass'
    df = df[df['PLAY TYPE'] == 'Pass']

    # Function to standardize zone names based on field position
    def normalize_zone(zone):
        if 'FLAT' in zone:
            return 'FLAT RIGHT' if 'RIGHT' in zone else 'FLAT LEFT'
        elif 'HOOK/CURL' in zone:
            return 'HOOK/CURL RIGHT' if 'RIGHT' in zone else 'HOOK/CURL LEFT'
        elif 'MIDDLE' in zone and 'DEEP' not in zone:
            return 'MIDDLE HOLE'
        return zone

    # Apply normalization to the PASS AREA column
    df['ZONE'] = df['PASS AREA'].apply(normalize_zone)

    # Group data by zone and calculate attempts and completions
    zone_stats = df.groupby('ZONE').agg(
        attempts=('RESULT', 'count'),
        completions=('RESULT', lambda x: (x == 'Complete').sum())
    )
    # Calculate success rate as completions divided by attempts
    zone_stats['success_rate'] = zone_stats['completions'] / zone_stats['attempts']

    # Define coordinates and dimensions for each zone on the field
    zone_map = {
        'FLAT LEFT': (0, 0, 10, 15),
        'HOOK/CURL LEFT': (10, 0, 10, 15),
        'MIDDLE HOLE': (20, 0, 13.3, 15),
        'HOOK/CURL RIGHT': (33.3, 0, 10, 15),
        'FLAT RIGHT': (43.3, 0, 10, 15),
        'DEEP LEFT': (0, 15, 17.7, 15),
        'DEEP MIDDLE': (17.7, 15, 17.7, 15),
        'DEEP RIGHT': (35.4, 15, 17.7, 15)
    }

    # Determine the maximum number of completions for color scaling
    max_completions = zone_stats['completions'].max() if not zone_stats.empty else 1

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 53.3)
    ax.set_ylim(0, 30)
    ax.set_facecolor('green')
    ax.set_title('Pass Completion Heatmap by Zone')

    # Draw each zone as a rectangle with color intensity based on completions
    for zone, (x, y, w, h) in zone_map.items():
        stats = zone_stats.loc[zone] if zone in zone_stats.index else {'completions': 0, 'attempts': 0, 'success_rate': 0.0}
        intensity = stats['completions'] / max_completions
        color = (1, 0, 0, intensity)  # Red with variable transparency
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='white', facecolor=color)
        ax.add_patch(rect)

        # Add zone label with completions, attempts, and success rate
        label = f"{zone}\n{int(stats['completions'])}/{int(stats['attempts'])}\n{stats['success_rate']:.0%}"
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', color='white', fontsize=8)

    # Add axis labels and remove grid
    ax.set_xlabel('Field Width (yards)')
    ax.set_ylabel('Field Length (yards)')
    ax.grid(False)

    # Display the plot in the Streamlit app
    st.pyplot(fig)
