import os
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from data_cleaner import preprocess_dataframe

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.markdown("""
<style>
    .stMetric { 
        border-left: 5px solid #2980b9; 
        padding: 15px; 
        border-radius: 8px; 
        background-color: transparent; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.2); 
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_pipeline():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, "models", "sentiment_model")
    
    if os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, batch_size=64)
    else:
        # Cloud Deployment Fallback: GitHub blocks the 250MB local model, so Streamlit must download a base model.
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", batch_size=64)

classifier = load_sentiment_pipeline()

st.title("Sentiment Analysis Dashboard")
st.markdown("Clean and preprocess review data, fine-tune a model, and run interactive visualizations.")

st.header("Batch Analytics")
data_source = st.radio("Choose Data Source:", ["Upload Files", "Use Local 'data' Folder"])

uploaded_files = []
local_file_path = None

if data_source == "Upload Files":
    uploaded_files = st.file_uploader("Upload Data (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"], accept_multiple_files=True)
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")
    if os.path.exists(data_dir):
        local_files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx', '.json'))]
        if local_files:
            selected_file = st.selectbox("Select a local dataset:", local_files)
            local_file_path = os.path.join(data_dir, selected_file)
        else:
            st.warning("No datasets found in the 'data' folder.")
    else:
        st.warning("'data' folder not found.")

files_to_process = uploaded_files if data_source == "Upload Files" else [local_file_path] if local_file_path else []

if files_to_process:
    if st.button("Execute Pipeline"):
        with st.spinner("Executing batch inference..."):
            all_dfs = []
            prog = st.progress(0)
            
            for i, file_obj in enumerate(files_to_process):
                is_local = isinstance(file_obj, str)
                file_name = os.path.basename(file_obj) if is_local else file_obj.name
                
                ext = file_name.split('.')[-1].lower()
                try:
                    if ext == 'csv':
                        df = pd.read_csv(file_obj, encoding='utf-8', on_bad_lines='skip')
                    elif ext == 'xlsx':
                        df = pd.read_excel(file_obj)
                    elif ext == 'json':
                        df = pd.read_json(file_obj)
                except UnicodeDecodeError:
                    if not is_local:
                        file_obj.seek(0)
                    df = pd.read_csv(file_obj, encoding='latin1', on_bad_lines='skip')
                    
                df_clean, _ = preprocess_dataframe(df, dataset_name=file_name)
                texts = [t[:512] for t in df_clean['cleaned_text'].tolist()]
                
                if texts:
                    results = []
                    batch_size = 64
                    inner_prog = st.progress(0, text=f"Analyzing {file_name}...")
                    
                    for j in range(0, len(texts), batch_size):
                        batch_results = classifier(texts[j:j+batch_size])
                        results.extend(batch_results)
                        inner_prog.progress(min(1.0, (j+batch_size)/len(texts)), text=f"Analyzing {file_name}...")
                    inner_prog.empty()
                    
                    df_clean['Sentiment'] = ['Positive' if r['label'] in ['POSITIVE', 'LABEL_1'] else 'Negative' for r in results]
                    df_clean['Confidence'] = [r['score'] for r in results]
                    df_clean['Text_Length'] = df_clean['cleaned_text'].apply(lambda x: len(str(x).split()))
                    all_dfs.append(df_clean)
                
                prog.progress((i+1)/len(files_to_process))
            
            if all_dfs:
                global_df = pd.concat(all_dfs, ignore_index=True)
                
                st.markdown("---")
                st.subheader("Executive Summary")
                total = len(global_df)
                pos = len(global_df[global_df['Sentiment'] == 'Positive'])
                neg = total - pos
                avg_conf = global_df['Confidence'].mean()
                
                col_total, col_pos, col_neg, col_avg = st.columns(4)
                col_total.metric("Total Documents", f"{total:,}")
                col_pos.metric("Positive", f"{pos:,}", f"{(pos/total)*100:.1f}%")
                col_neg.metric("Negative", f"{neg:,}", f"-{(neg/total)*100:.1f}%")
                col_avg.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
                
                st.markdown("---")
                st.subheader("Interactive Analytics")
                
                col_donut, col_box = st.columns(2)
                
                with col_donut:
                    fig_donut = px.pie(global_df, names='Sentiment', hole=0.5, 
                                       color='Sentiment', color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'},
                                       title="Global Sentiment Distribution")
                    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                    fig_donut.update_layout(margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_donut, use_container_width=True)
                    
                with col_box:
                    fig_box = px.box(global_df, x="Dataset_Name", y="Confidence", color="Sentiment",
                                     color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'},
                                     title="Confidence Spread by Dataset")
                    fig_box.update_layout(margin=dict(t=40, b=0, l=0, r=0))
                    st.plotly_chart(fig_box, use_container_width=True)

                st.markdown("---")
                st.subheader("Behavioral & Text Length Analytics")
                fig_hist = px.histogram(global_df, x="Text_Length", color="Sentiment", marginal="violin",
                                        color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'},
                                        title="Text Length Density by Sentiment")
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("---")
                st.subheader("Top Impactful Reviews (Extremes Analysis)")
                top_pos = global_df[global_df['Sentiment']=='Positive'].nlargest(3, 'Confidence')
                top_neg = global_df[global_df['Sentiment']=='Negative'].nlargest(3, 'Confidence')
                
                c_pos, c_neg = st.columns(2)
                with c_pos:
                    st.success("**Top Positive Reviews**")
                    for _, row in top_pos.iterrows():
                        st.write(f"🟢 ({row['Confidence']*100:.1f}%) {row['cleaned_text']}")
                with c_neg:
                    st.error("**Top Negative Reviews**")
                    for _, row in top_neg.iterrows():
                        st.write(f"🔴 ({row['Confidence']*100:.1f}%) {row['cleaned_text']}")

                st.markdown("---")
                st.subheader("Lexical Theme Extraction")
                
                pos_text = " ".join(global_df[global_df['Sentiment'] == 'Positive']['cleaned_text'])
                neg_text = " ".join(global_df[global_df['Sentiment'] == 'Negative']['cleaned_text'])
                
                col_wc_pos, col_wc_neg = st.columns(2)
                with col_wc_pos:
                    if len(pos_text.strip()) > 0:
                        try:
                            wc = WordCloud(width=600, height=300, background_color='white', colormap='Greens').generate(pos_text)
                            fig, ax = plt.subplots()
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        except ValueError: pass
                with col_wc_neg:
                    if len(neg_text.strip()) > 0:
                        try:
                            wc = WordCloud(width=600, height=300, background_color='white', colormap='Reds').generate(neg_text)
                            fig, ax = plt.subplots()
                            ax.imshow(wc, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        except ValueError: pass

                st.markdown("---")
                st.subheader("Database Explorer & Export")
                
                filter_sent = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative"])
                display_df = global_df if filter_sent == "All" else global_df[global_df['Sentiment'] == filter_sent]
                
                st.dataframe(display_df[['Dataset_Name', 'cleaned_text', 'Sentiment', 'Confidence', 'Text_Length']].head(50))
                
                csv = global_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Report (CSV)", data=csv, file_name='sentiment_report.csv', mime='text/csv')
            else:
                st.warning("No valid text data could be parsed from the uploaded files.")
