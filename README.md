# 🧠 Enterprise Sentiment Intelligence Platform

A production-ready, highly interactive Sentiment Analysis Dashboard built using Python, Streamlit, Plotly, and Hugging Face Transformers. This application leverages deep learning (DistilBERT) to dynamically ingest, analyze, and visualize customer sentiment across heterogeneous datasets.

## 🚀 Features

* **Multi-Format Ingestion:** Seamlessly upload and process `.csv`, `.xlsx`, and `.json` files.
* **Fault-Tolerant Engine:** Built-in safeguards (`on_bad_lines`) and encoding fallbacks (`utf-8` to `latin1`) prevent pipeline crashes from corrupted data.
* **Dynamic Schema Detection:** Automatically identifies the unstructured text column regardless of the dataset's native column headers.
* **Advanced Analytics Suite:** 
  * **Interactive Plotly Visualizations:** Normalized Stacked Bar charts, Confidence Spread Boxplots, and Text Length Density Violins.
  * **Automated Anomaly Detection:** Real-time algorithmic tracking of negative sentiment thresholds to trigger operational warnings.
  * **Lexical Theme Extraction:** Automated generation of Positive and Negative Word Clouds.
* **Database Explorer:** Filter, analyze, and instantly export parsed insights to CSV.

## 📂 Project Architecture

```text
Sentiment_Analysis/
│
├── data/                    # Storage for raw and processed datasets
│   ├── sample_reviews.csv
│   └── quick_test_reviews.csv
│
├── models/                  # Custom trained Machine Learning weights
│   └── sentiment_model/     # Fine-Tuned DistilBERT architecture
│
├── src/                     # Source Code Modules
│   ├── app.py               # Streamlit application & visualization engine
│   ├── data_cleaner.py      # Text sanitization and preprocessing pipeline
│   └── model_trainer.py     # Hugging Face training/fine-tuning script
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   Navigate to your desired directory and initialize the project.
   
2. **Activate the Virtual Environment:**
   (Assuming the `venv` is already built in your workspace)
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   If you are setting this up on a new machine, install the required packages:
   ```powershell
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Launching the Dashboard
To start the Enterprise Analytics Dashboard, run the following command from the root directory:
```powershell
python -m streamlit run src/app.py
```
This will open the dashboard in your default browser at `http://localhost:8501`.

### 2. Fine-Tuning the Model
If you wish to re-train or fine-tune the core NLP engine on your own specific domain data, you can execute the trainer script. It will automatically save the new weights into the `models/` directory.
```powershell
python src/model_trainer.py
```

## 📊 Analytics Breakdown

* **Real-Time Inference:** Paste any unstructured text into the portal to receive instantaneous AI classification and Neural Confidence scoring.
* **Enterprise Batch Analytics:** Upload thousands of rows. The engine will batch process them in chunks of 64, providing a visual progress bar. Once completed, you will be presented with a comprehensive executive summary outlining overall brand health.

## ⚙️ Technologies Used
* **Frontend:** Streamlit
* **Data Manipulation:** Pandas, NumPy
* **NLP & Deep Learning:** Hugging Face `transformers`, PyTorch (`torch`)
* **Visualizations:** Plotly Express, Matplotlib, WordCloud
