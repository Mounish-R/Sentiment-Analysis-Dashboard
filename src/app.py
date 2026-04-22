import os
import torch
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from data_cleaner import clean_text, preprocess_dataframe

# Configuration
st.set_page_config(page_title="Enterprise NLP Analytics", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    .stMetric { 
        border-left: 5px solid #2980b9; 
        padding: 15px; 
        border-radius: 8px; 
        background-color: transparent; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.2); 
    }
    h1, h2, h3 { font-family: 'Inter', 'Segoe UI', sans-serif; font-weight: 700; }
    .stButton>button { background-color: #2980b9; color: white; border-radius: 6px; padding: 0.6rem 2rem; border: none; font-weight: 600; }
    .stButton>button:hover { background-color: #3498db; color: white; border: 1px solid #2980b9; }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧠 AI Control Center")
st.sidebar.markdown("Configure NLP engine parameters.")

model_choice = st.sidebar.selectbox("NLP Architecture", ["Fine-Tuned DistilBERT", "HuggingFace Default (SST-2)"])

@st.cache_resource
def load_sentiment_pipeline(choice):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, "models", "sentiment_model")
    
    if choice == "Fine-Tuned DistilBERT" and os.path.exists(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, batch_size=64)
    return pipeline("sentiment-analysis", batch_size=64)

classifier = load_sentiment_pipeline(model_choice)
st.sidebar.success("Hardware Accel: CPU")

# Main Application
st.title("📊 Advanced Sentiment Intelligence")
st.markdown("Automated NLP processing for heterogeneous datasets with anomaly detection and interactive Plotly visualization.")

tab_single, tab_batch = st.tabs(["🔍 Real-Time Inference", "📂 Enterprise Batch Analytics"])

with tab_single:
    st.header("Real-Time NLP Inference")
    user_input = st.text_area("Input unstructured text:", "The integration was flawless, but the documentation is extremely poor.", height=100)
    
    if st.button("Run AI Prediction"):
        with st.spinner("Processing through neural network..."):
            cleaned = clean_text(user_input)
            result = classifier(cleaned[:512])[0]
            
            col_sentiment, col_confidence = st.columns(2)
            sentiment = 'POSITIVE' if result['label'] in ['POSITIVE', 'LABEL_1'] else 'NEGATIVE'
            color = 'green' if sentiment == 'POSITIVE' else 'red'
            
            col_sentiment.markdown(f"<h3 style='color: {color};'>Classification: {sentiment}</h3>", unsafe_allow_html=True)
            col_confidence.metric("Neural Confidence", f"{result['score']*100:.2f}%")
            
            st.info(f"**Sanitized Input passed to model:** {cleaned}")

with tab_batch:
    st.header("Multi-Format Batch Ingestion")
    uploaded_files = st.file_uploader("Upload Data (CSV, XLSX, JSON)", type=["csv", "xlsx", "json"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Execute Global NLP Pipeline"):
            with st.spinner("Executing batch inference. Mapping schemas and normalizing data..."):
                all_dfs = []
                prog = st.progress(0)
                
                for i, file in enumerate(uploaded_files):
                    ext = file.name.split('.')[-1].lower()
                    try:
                        if ext == 'csv':
                            df = pd.read_csv(file, encoding='utf-8', on_bad_lines='skip')
                        elif ext == 'xlsx':
                            df = pd.read_excel(file)
                        elif ext == 'json':
                            df = pd.read_json(file)
                    except UnicodeDecodeError:
                        file.seek(0)
                        df = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
                        
                    df_clean, _ = preprocess_dataframe(df, dataset_name=file.name)
                    texts = [t[:512] for t in df_clean['cleaned_text'].tolist()]
                    
                    if texts:
                        results = []
                        batch_size = 64
                        inner_prog = st.progress(0, text=f"Analyzing {file.name}...")
                        
                        for j in range(0, len(texts), batch_size):
                            batch_results = classifier(texts[j:j+batch_size])
                            results.extend(batch_results)
                            inner_prog.progress(min(1.0, (j+batch_size)/len(texts)), text=f"Analyzing {file.name}...")
                        inner_prog.empty()
                        
                        df_clean['Sentiment'] = ['Positive' if r['label'] in ['POSITIVE', 'LABEL_1'] else 'Negative' for r in results]
                        df_clean['Confidence'] = [r['score'] for r in results]
                        df_clean['Text_Length'] = df_clean['cleaned_text'].apply(lambda x: len(str(x).split()))
                        all_dfs.append(df_clean)
                    
                    prog.progress((i+1)/len(uploaded_files))
                
                if all_dfs:
                    global_df = pd.concat(all_dfs, ignore_index=True)
                    
                    st.markdown("---")
                    st.subheader("📈 Executive NLP Summary")
                    total = len(global_df)
                    pos = len(global_df[global_df['Sentiment'] == 'Positive'])
                    neg = total - pos
                    avg_conf = global_df['Confidence'].mean()
                    
                    col_total, col_pos, col_neg, col_avg = st.columns(4)
                    col_total.metric("Total Documents", f"{total:,}")
                    col_pos.metric("Positive Sentiment", f"{pos:,}", f"{(pos/total)*100:.1f}%")
                    col_neg.metric("Negative Sentiment", f"{neg:,}", f"-{(neg/total)*100:.1f}%")
                    col_avg.metric("Avg AI Confidence", f"{avg_conf*100:.1f}%")
                    
                    st.markdown("---")
                    st.subheader("📊 Deep Interactive Analytics")
                    
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
                    col_stack, col_hist = st.columns(2)
                    
                    with col_stack:
                        cross_tab = pd.crosstab(global_df['Dataset_Name'], global_df['Sentiment'], normalize='index') * 100
                        cross_tab = cross_tab.reset_index()
                        
                        fig_stack = px.bar(cross_tab, x="Dataset_Name", y=cross_tab.columns[1:], 
                                           title="Normalized Proportion (%) by Dataset",
                                           color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'})
                        fig_stack.update_layout(barmode='stack', yaxis_title="Percentage %")
                        st.plotly_chart(fig_stack, use_container_width=True)

                    with col_hist:
                        fig_hist = px.histogram(global_df, x="Text_Length", color="Sentiment", marginal="violin",
                                                color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'},
                                                title="Text Length Density & Distribution")
                        st.plotly_chart(fig_hist, use_container_width=True)

                    st.markdown("---")
                    st.subheader("🚨 Automated Insights & Anomaly Detection")
                    
                    neg_ratio = neg / total
                    if neg_ratio > 0.4:
                        st.error(f"**ANOMALY DETECTED:** High volume of negative sentiment ({neg_ratio*100:.1f}%). Immediate review of operational/product pipelines is recommended.")
                    elif pos > neg:
                        st.success(f"**POSITIVE TREND:** Customer satisfaction is trending positively at {(pos/total)*100:.1f}%.")
                    
                    st.markdown("**Top Impactful Reviews (Highest AI Confidence)**")
                    top_pos = global_df[global_df['Sentiment']=='Positive'].nlargest(1, 'Confidence')
                    top_neg = global_df[global_df['Sentiment']=='Negative'].nlargest(1, 'Confidence')
                    
                    if not top_pos.empty:
                        st.info(f"🟢 **Highest Confidence Positive ({top_pos['Confidence'].values[0]*100:.1f}%):** {top_pos['cleaned_text'].values[0]}")
                    if not top_neg.empty:
                        st.warning(f"🔴 **Highest Confidence Negative ({top_neg['Confidence'].values[0]*100:.1f}%):** {top_neg['cleaned_text'].values[0]}")

                    st.markdown("---")
                    st.subheader("☁️ Lexical Theme Extraction")
                    
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
                    st.subheader("🗄️ Database Explorer & Export")
                    
                    filter_sent = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative"])
                    display_df = global_df if filter_sent == "All" else global_df[global_df['Sentiment'] == filter_sent]
                    
                    st.dataframe(display_df[['Dataset_Name', 'cleaned_text', 'Sentiment', 'Confidence', 'Text_Length']].head(50))
                    
                    csv = global_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Enterprise Report (CSV)", data=csv, file_name='enterprise_sentiment_report.csv', mime='text/csv')
                else:
                    st.warning("No valid text data could be parsed from the uploaded files.")
