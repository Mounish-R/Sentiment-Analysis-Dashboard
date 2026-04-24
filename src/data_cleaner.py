import re
import pandas as pd

def detect_text_column(df):
    possible_names = ['text', 'review', 'comment', 'sentence']
    for col in df.columns:
        if str(col).lower().strip() in possible_names:
            return col
    return df.columns[0]

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_dataframe(df, dataset_name="dataset"):
    df_cleaned = df.copy()
    text_column = detect_text_column(df_cleaned)
    
    df_cleaned[text_column] = df_cleaned[text_column].fillna("").astype(str)
    df_cleaned['cleaned_text'] = df_cleaned[text_column].apply(clean_text)
    
    df_cleaned = df_cleaned[df_cleaned['cleaned_text'] != ""]
    df_cleaned['Dataset_Name'] = dataset_name
    
    return df_cleaned, text_column
