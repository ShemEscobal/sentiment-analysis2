import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fb_classifier import classify_emotions

def main():
    st.title("Emotion Classification App")
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
                
                # Display results as pie chart
                st.write("Prediction Results:")
                fig, ax = plt.subplots()
                wedges, texts, autotexts = ax.pie(
                    label_counts.values(), 
                    autopct='%1.1f%%', 
                    startangle=90,
                    pctdistance=0.85
                )
                # Remove labels and show only percentages
                for text in texts:
                    text.set_visible(False)
                st.pyplot(fig)


                # Display predictions in a table with counts
                prediction_df = pd.DataFrame({'Predicted Emotion': predictions})
                count_df = prediction_df.value_counts().reset_index(name='Count')
                st.dataframe(count_df)
            else:
                st.error("No column selected.")
    
if __name__ == "__main__":
    main()
