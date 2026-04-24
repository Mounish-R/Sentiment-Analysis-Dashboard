import os
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from data_cleaner import preprocess_dataframe

# Must be the very first Streamlit command
st.set_page_config(page_title="Enterprise Sentiment Intelligence", page_icon="📈", layout="wide")

# Advanced CSS Injection for a Premium Look
st.markdown("""
<style>
    /* Main Background & Font Tweaks */
    .reportview-container {
        background: #f8f9fa;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #2c3e50;
    }
    /* Metric Cards Styling */
    div[data-testid="metric-container"] {
        background-color: white;
        border-left: 5px solid #3498db;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
    /* Button Styling */
    .stButton>button {
        width: 100%;
        background-color: #2980b9;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #3498db;
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.4);
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
        # Cloud Deployment Fallback
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", batch_size=64)

classifier = load_sentiment_pipeline()

# --- HERO SECTION ---
st.title("📈 Enterprise Sentiment Intelligence Platform")
st.markdown("<p style='font-size: 1.2rem; color: #7f8c8d;'>A centralized machine learning hub for deep-text sentiment analytics, behavioral extraction, and executive reporting.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103285.png", width=60) # Placeholder generic AI icon
    st.title("Data Ingestion Engine")
    st.markdown("Select your input vector to begin batch processing.")
    
    data_source = st.radio("Primary Data Source:", ["Local Database", "Upload External Files"])
    
    uploaded_files = []
    local_file_path = None
    
    if data_source == "Upload External Files":
        uploaded_files = st.file_uploader("Drop CSV, XLSX, or JSON here", type=["csv", "xlsx", "json"], accept_multiple_files=True)
    else:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, "data")
        if os.path.exists(data_dir):
            local_files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.xlsx', '.json'))]
            if local_files:
                selected_file = st.selectbox("Select local index:", local_files)
                local_file_path = os.path.join(data_dir, selected_file)
            else:
                st.warning("No indices found in the 'data' partition.")
        else:
            st.error("System error: 'data' partition unlinked.")
            
    files_to_process = uploaded_files if data_source == "Upload External Files" else [local_file_path] if local_file_path else []
    
    st.markdown("<br>", unsafe_allow_html=True)
    execute_btn = st.button("🚀 Initialize Pipeline")

# --- MAIN EXECUTION BLOCK ---
if files_to_process and execute_btn:
    with st.spinner("Neural Engine Active: Parsing and Inferencing..."):
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
                inner_prog = st.progress(0, text=f"Vectorizing {file_name}...")
                
                for j in range(0, len(texts), batch_size):
                    batch_results = classifier(texts[j:j+batch_size])
                    results.extend(batch_results)
                    inner_prog.progress(min(1.0, (j+batch_size)/len(texts)), text=f"Vectorizing {file_name}...")
                inner_prog.empty()
                
                df_clean['Sentiment'] = ['Positive' if r['label'] in ['POSITIVE', 'LABEL_1'] else 'Negative' for r in results]
                df_clean['Confidence'] = [r['score'] for r in results]
                df_clean['Text_Length'] = df_clean['cleaned_text'].apply(lambda x: len(str(x).split()))
                all_dfs.append(df_clean)
            
            prog.progress((i+1)/len(files_to_process))
        
        prog.empty()
        
        if all_dfs:
            global_df = pd.concat(all_dfs, ignore_index=True)
            
            # --- 1. EXECUTIVE SUMMARY ---
            st.subheader("📊 Executive Summary")
            total = len(global_df)
            pos = len(global_df[global_df['Sentiment'] == 'Positive'])
            neg = total - pos
            avg_conf = global_df['Confidence'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Datapoints Processed", f"{total:,}")
            col2.metric("Overall Positive Sentiment", f"{pos:,}", f"{(pos/total)*100:.1f}%")
            col3.metric("Overall Negative Sentiment", f"{neg:,}", f"-{(neg/total)*100:.1f}%")
            col4.metric("AI Confidence Average", f"{avg_conf*100:.1f}%")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- 2. ADVANCED INTERACTIVE ANALYTICS ---
            st.subheader("📈 Core Analytics Engine")
            col_donut, col_box = st.columns(2)
            
            with col_donut:
                fig_donut = px.pie(global_df, names='Sentiment', hole=0.45, 
                                   color='Sentiment', color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'})
                fig_donut.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
                fig_donut.update_layout(title_text="Global Sentiment Distribution", title_x=0.5, margin=dict(t=50, b=20, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_donut, use_container_width=True)
                
            with col_box:
                fig_box = px.box(global_df, x="Dataset_Name", y="Confidence", color="Sentiment",
                                 color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'})
                fig_box.update_layout(title_text="Confidence Spread by Sub-Index", title_x=0.5, margin=dict(t=50, b=20, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_box, use_container_width=True)

            # --- 3. BEHAVIORAL ANALYTICS ---
            st.markdown("---")
            st.subheader("🧠 Behavioral & Structural Analytics")
            fig_hist = px.histogram(global_df, x="Text_Length", color="Sentiment", marginal="violin",
                                    color_discrete_map={'Positive':'#2ecc71', 'Negative':'#e74c3c'}, opacity=0.8)
            fig_hist.update_layout(title_text="Text Length Density & User Verbosity Profiling", title_x=0.5, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_hist, use_container_width=True)

            # --- 4. EXTREMES ANALYSIS ---
            st.markdown("---")
            st.subheader("🚨 Extreme Polarized Extractions")
            st.markdown("The neural network's absolute highest-confidence extractions across the global dataset.")
            top_pos = global_df[global_df['Sentiment']=='Positive'].nlargest(3, 'Confidence')
            top_neg = global_df[global_df['Sentiment']=='Negative'].nlargest(3, 'Confidence')
            
            c_pos, c_neg = st.columns(2)
            with c_pos:
                st.success("**🏆 Top Positive Extractions**")
                for _, row in top_pos.iterrows():
                    st.write(f"**Confidence: {row['Confidence']*100:.1f}%**")
                    st.caption(f"_{row['cleaned_text']}_")
                    st.markdown("---")
            with c_neg:
                st.error("**⚠️ Top Negative Extractions**")
                for _, row in top_neg.iterrows():
                    st.write(f"**Confidence: {row['Confidence']*100:.1f}%**")
                    st.caption(f"_{row['cleaned_text']}_")
                    st.markdown("---")

            # --- 5. LEXICAL THEME EXTRACTION ---
            st.markdown("---")
            st.subheader("☁️ Lexical Theme Generation")
            pos_text = " ".join(global_df[global_df['Sentiment'] == 'Positive']['cleaned_text'])
            neg_text = " ".join(global_df[global_df['Sentiment'] == 'Negative']['cleaned_text'])
            
            col_wc_pos, col_wc_neg = st.columns(2)
            with col_wc_pos:
                if len(pos_text.strip()) > 0:
                    try:
                        wc = WordCloud(width=800, height=400, background_color='white', colormap='summer').generate(pos_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title("Positive Lexicon", fontsize=16, color="#2ecc71", pad=20)
                        st.pyplot(fig)
                    except ValueError: pass
            with col_wc_neg:
                if len(neg_text.strip()) > 0:
                    try:
                        wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(neg_text)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        ax.set_title("Negative Lexicon", fontsize=16, color="#e74c3c", pad=20)
                        st.pyplot(fig)
                    except ValueError: pass

            # --- 6. DATABASE EXPLORER ---
            st.markdown("---")
            with st.expander("🗄️ Open Global Database Explorer & Export Module"):
                filter_sent = st.selectbox("Filter Database by Neural Classification", ["All", "Positive", "Negative"])
                display_df = global_df if filter_sent == "All" else global_df[global_df['Sentiment'] == filter_sent]
                
                st.dataframe(display_df[['Dataset_Name', 'cleaned_text', 'Sentiment', 'Confidence', 'Text_Length']].head(100), use_container_width=True)
                
                csv = global_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Master Report (CSV)", data=csv, file_name='enterprise_sentiment_report.csv', mime='text/csv')
        else:
            st.warning("No valid text data could be parsed from the selected sources.")
elif not files_to_process:
    st.info("👈 Please select a data source from the control panel on the left to begin.")
