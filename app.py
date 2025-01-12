import os
import streamlit as st
import pandas as pd
from fb_classifier import classify_emotions

# Set upload folder using absolute path
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Streamlit app
st.title("Feedback Classification App")

# File uploader
uploaded_file = st.file_uploader("Upload an Excel file", type="xlsx")

if uploaded_file is not None:
    try:
        # Save the uploaded file
        filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Read the uploaded Excel file
        df = pd.read_excel(filepath)
        
        # Display columns for selection
        selected_column = st.selectbox("Select the column to analyze", df.columns)
        
        if st.button("Classify Emotions"):
            # Classify emotions
            classified_df = classify_emotions(df, selected_column)
            
            # Prepare predictions
            predictions = classified_df['predicted_emotion'].tolist()
            
            # Count labels
            label_counts = {}
            for label in predictions:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            total_responses = len(predictions)
            
            # Display results
            st.write("### Classification Results")
            st.write(f"Total Responses: {total_responses}")
            st.write("Label Counts:")
            st.write(label_counts)
            
            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except Exception as e:
                st.warning(f"Failed to remove temporary file: {str(e)}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
