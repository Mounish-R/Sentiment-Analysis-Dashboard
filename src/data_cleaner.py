import re
import pandas as pd

def detect_text_column(df):
    """
    Dynamically detect the text column based on common naming conventions.
    
    System Design & Flexibility:
    - This function allows the system to seamlessly ingest heterogeneous datasets.
    - Instead of breaking or demanding manual user input, it smartly detects 'text', 'review', etc.
    - Fallback mechanism ensures that structurally ambiguous CSVs still process gracefully.
    """
    possible_names = ['text', 'review', 'comment', 'sentence']
    for col in df.columns:
        if str(col).lower().strip() in possible_names:
            return col
    # Fallback to the first column if no common names match
    return df.columns[0]

def clean_text(text):
    """
    Basic text cleaning for review/social data.
    
    Real-world Usability:
    - Safely handles unexpected types (floats, None) by coercing to strings.
    - Removes HTML, URLs, and social media noise which usually confuse NLP models.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet/social data
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df, dataset_name="dataset"):
    """
    Preprocess a pandas DataFrame containing text data.
    
    Design Improvements:
    - Encapsulates error handling for missing values (NaN to empty string).
    - Adds data provenance tracking by appending a 'Dataset_Name' column, which is vital 
      for downstream multi-dataset grouped analytics.
    """
    df_cleaned = df.copy()
    
    text_column = detect_text_column(df_cleaned)
    
    # Convert all inputs to string and handle nulls
    df_cleaned[text_column] = df_cleaned[text_column].fillna("").astype(str)
    
    # Apply cleaning
    df_cleaned['cleaned_text'] = df_cleaned[text_column].apply(clean_text)
    
    # Filter out empty strings to prevent wasteful inference calls
    df_cleaned = df_cleaned[df_cleaned['cleaned_text'] != ""]
    
    # Add dataset origin tracking
    df_cleaned['Dataset_Name'] = dataset_name
    
    return df_cleaned, text_column
