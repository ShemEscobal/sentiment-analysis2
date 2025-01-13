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
                
                # Define color mapping for emotions
                emotion_colors = {
                    'approval': '#1f77b4',
                    'admiration': '#ff7f0e',
                    'neutral': '#2ca02c',
                    'caring': '#d62728',
                    'realization': '#9467bd',
                    'joy': '#8c564b',
                    'NO_TEXT': '#e377c2',
                    'gratitude': '#7f7f7f',
                    'curiosity': '#bcbd22'
                }



                # Calculate percentages
                total = sum(label_counts.values())
                percentages = {k: (v/total)*100 for k, v in label_counts.items()}
                
                # Create table with colors and percentages
                table_data = []
                for emotion, count in label_counts.items():
                    color = emotion_colors.get(emotion, '#000000')
                    table_data.append({
                        'Predicted Emotion': emotion,
                        'Count': count,
                        'Percentage': f"{percentages[emotion]:.1f}%",
                        'Color': color
                    })
                
                # Display pie chart with only colors
                st.write("Prediction Results:")
                fig, ax = plt.subplots()
                wedges, texts, autotexts = ax.pie(
                    label_counts.values(),
                    colors=[emotion_colors.get(emotion, '#000000') for emotion in label_counts.keys()],
                    startangle=90
                )
                # Remove all labels and percentages
                for text in texts:
                    text.set_visible(False)
                for autotext in autotexts:
                    autotext.set_visible(False)
                st.pyplot(fig)

                # Display styled table
                st.dataframe(
                    pd.DataFrame(table_data),
                    column_config={
                        "Color": st.column_config.ColorColumn(
                            "Color",
                            help="Color representation of the emotion",
                            required=True
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )


            else:
                st.error("No column selected.")
    
if __name__ == "__main__":
    main()
