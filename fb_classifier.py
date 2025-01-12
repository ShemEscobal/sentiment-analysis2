import pandas as pd
from transformers import pipeline
import torch

def classify_emotions(df, text_column):
    """
    Classify emotions in a given DataFrame using the RoBERTa Go Emotions model.
    
    Parameters:
    - df: DataFrame containing the text to classify
    - text_column: Name of the column containing text to classify
    
    Returns:
    - DataFrame with predicted emotions
    """
    try:
        # Explicitly set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the emotion classification pipeline with device specification
        emotion_classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            device=device
        )
        
        # Validate that the text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
        
        def preprocess_text(text):
            """Clean and validate text input."""
            # Convert to string if not already
            if not isinstance(text, str):
                text = str(text)
            
            # Handle NaN, None, empty strings and whitespace
            if pd.isna(text) or text.strip() == "":
                return ""
            return text.strip()
        
        def classify_batch(texts):
            """
            Classify emotions for a batch of texts with proper handling of invalid inputs.
            """
            processed_texts = [preprocess_text(text) for text in texts]
            results = []
            
            for text in processed_texts:
                if text == "":
                    results.append({"label": "NO_TEXT"})
                else:
                    try:
                        # Get prediction for valid text with explicit truncation
                        prediction = emotion_classifier(text, max_length=512, truncation=True)[0]
                        results.append(prediction)
                    except Exception as e:
                        print(f"Error processing text: {str(e)}")
                        results.append({"label": "ERROR"})
            
            return [result['label'] for result in results]

        # Get the texts to classify
        texts_to_classify = df[text_column].fillna("").tolist()
        
        # Process in smaller batches to avoid memory issues
        batch_size = 32
        all_predictions = []
        
        for i in range(0, len(texts_to_classify), batch_size):
            batch = texts_to_classify[i:i + batch_size]
            predictions = classify_batch(batch)
            all_predictions.extend(predictions)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'predicted_emotion': all_predictions
        })
        
        return results_df
        
    except Exception as e:
        print(f"Error in emotion classification: {str(e)}")
        raise  # Re-raise the exception for better error handling

# Example usage:
if __name__ == "__main__":
    # Create sample DataFrame
    sample_df = pd.DataFrame({
        'text': ['I am happy today!', 'This makes me angry', 'I feel sad']
    })
    
    try:
        results = classify_emotions(sample_df, 'text')
        print(results)
    except Exception as e:
        print(f"Failed to classify emotions: {str(e)}")
