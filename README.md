# 🧠 Sentiment Analysis Dashboard

A complete, end-to-end Machine Learning web application designed to clean review data, fine-tune a Hugging Face transformer model, visualize predictive insights, and deploy an interactive dashboard using Streamlit. Built using Python, PyTorch, and Plotly, this platform demonstrates a complete NLP lifecycle from raw data to actionable executive intelligence.

## 🚀 Key Features

* **Dual-Source Ingestion:** Seamlessly upload `.csv`, `.xlsx`, and `.json` files via the browser, or directly select pre-existing datasets from the local `data/` folder.
* **Local Model Fine-Tuning:** Natively downloads, tokenizes, trains, and saves a customized `distilbert-base-uncased` transformer model entirely from scratch using the Hugging Face Trainer API.
* **Behavioral Analytics:** Features a Plotly Histogram and Marginal Violin Plot to analyze the psychological correlation between user sentiment and verbosity (Text Length Density).
* **Extremes Analysis:** Automatically extracts and highlights the absolute highest-confidence positive and negative reviews, giving an immediate pulse on polarized customer feedback.
* **Interactive Visual Insights:** Generates dynamic Plotly Donut Charts for sentiment distribution, Confidence Spread Boxplots, and lexical Word Clouds.

## 📂 Project Architecture

```text
Sentiment_Analysis/
│
├── data/                    # Storage for raw and processed datasets (e.g., Twitter.csv)
├── models/                  # Custom trained Machine Learning weights
│   └── sentiment_model/     # Local Fine-Tuned DistilBERT architecture (Git-ignored)
│
├── src/                     # Source Code Modules
│   ├── app.py               # Streamlit application & batch visualization engine
│   ├── data_cleaner.py      # Text sanitization and preprocessing pipeline
│   └── model_trainer.py     # Hugging Face training/fine-tuning script
│
├── requirements.txt         # Project dependencies
├── Project_Abstract.md      # Detailed academic/professional write-up
└── README.md                # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository:**
   Navigate to your desired directory and clone the project.
   
2. **Activate the Virtual Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Hugging Face Authentication (Required for faster downloads):**
   It is highly recommended to log into Hugging Face to prevent rate-limiting during model downloads.
   ```powershell
   hf auth login
   ```
   *(Paste your Access Token when prompted)*

## 💻 Usage Pipeline

This project is strictly aligned to a complete ML pipeline. **You must train the model before launching the dashboard.**

### Step 1: Fine-Tune the Base Model
Execute the training script. This script downloads the base `distilbert` model, loads the IMDb dataset, tokenizes the text, trains the model for 1 epoch, and saves the customized weights into the `models/` directory.
```powershell
python src/model_trainer.py
```

### Step 2: Launch the Batch Analytics Dashboard
Once the training terminal says "Training complete!", you can boot up the interactive interface:
```powershell
python -m streamlit run src/app.py
```
This will open the dashboard in your default browser. You can immediately select a local dataset from your `data/` folder or upload a new one to begin batch processing!

## ☁️ Streamlit Cloud Deployment

This project is configured for **Continuous Deployment (CD)** via Streamlit Community Cloud.
Because the fine-tuned `models/` folder is over 250MB, it is protected by `.gitignore` and is not pushed to GitHub. To ensure the cloud application doesn't crash, `src/app.py` contains a specific **Cloud Deployment Fallback** that automatically downloads a pre-finetuned model directly from the Hugging Face Hub if it detects it is running in a server environment.

## ⚙️ Technologies Used
* **Frontend:** Streamlit
* **Deep Learning Framework:** PyTorch (`torch`)
* **Natural Language Processing:** Hugging Face `transformers` (Trainer API, DistilBERT), `datasets`
* **Data Engineering:** Pandas, NumPy
* **Data Visualization:** Plotly Express, Matplotlib, WordCloud
