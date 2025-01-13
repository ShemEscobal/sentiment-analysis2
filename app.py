import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fb_classifier import classify_emotions
import colorsys
import random

def generate_colors(num_colors):
    """Generate a list of distinct colors using the HSL color space"""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = 0.5
        saturation = 0.7
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}')
    return colors

def get_color_mapping(labels):
    """Create a color mapping for the given labels"""
    unique_labels = sorted(list(set(labels)))
    colors = generate_colors(len(unique_labels))
    return dict(zip(unique_labels, colors))

def main():
    st.title("Feedback Classification App")
    st.subheader("Upload an Excel file with text data and select a column you want to classify.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        # Read the uploaded Excel file
        df = pd.read_excel(uploaded_file)
        
        # Validate data size
        if len(df) > 1000:
            st.error("File too large. Please limit to 1000 rows.")
            return
        
        # Get the columns for selection
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select a column for prediction", columns)
        
        if st.button("Classify Emotions"):
            if selected_column:
                # Process in smaller batches
                batch_size = 50
                results = []
                
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i + batch_size].copy()
                    classified_batch = classify_emotions(batch, selected_column)
                    results.append(classified_batch['predicted_emotion'])
                
                # Combine results
                predictions = pd.concat(results).tolist()
                
                # Count labels
                label_counts = {}
                for label in predictions:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                # Generate color mapping
                color_mapping = get_color_mapping(label_counts.keys())

                # Calculate percentages
                total = sum(label_counts.values())
                percentages = {k: (v/total)*100 for k, v in label_counts.items()}
                
                # Create table with colors and percentages
                table_data = []
                for emotion, count in label_counts.items():
                    color = color_mapping.get(emotion, '#000000')
                    table_data.append({
                        'Predicted Emotion': emotion,
                        'Count': count,
                        'Percentage': f"{percentages[emotion]:.1f}%",
                        'Color': color
                    })

                # Display pie chart with only colors
                st.write("Prediction Results:")
                fig, ax = plt.subplots()
                ax.pie(
                    label_counts.values(),
                    colors=[color_mapping.get(emotion, '#000000') for emotion in label_counts.keys()],
                    startangle=90
                )
                st.pyplot(fig)

                # Display styled table with color indicators
                def color_cell(val):
                    return f'background-color: {val}; color: {val}'  # Set text color to match background
                
                # Sort by percentage (descending)
                table_df = pd.DataFrame(table_data)
                table_df['PercentageValue'] = table_df['Percentage'].str.rstrip('%').astype(float)
                table_df = table_df.sort_values('PercentageValue', ascending=False)
                
                # Apply styling and display
                styled_df = table_df[['Predicted Emotion', 'Count', 'Percentage', 'Color']].style.applymap(color_cell, subset=['Color'])
                st.dataframe(
                    styled_df,
                    hide_index=True,
                    use_container_width=True
                )

            else:
                st.error("No column selected.")
    
if __name__ == "__main__":
    main()
