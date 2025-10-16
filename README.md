# ⚡ DataForge: Automated Data Cleaning, Visualization & ML Pipeline

**DataForge** is a **Streamlit-powered machine learning workflow tool** that simplifies the complete data journey — from cleaning and exploration to visualization, model training, and report generation. It is designed to be **user-friendly, professional, and fully automated**, making it ideal for data scientists, analysts, and students.

## 🚀 Features

### 🧹 Data Cleaning

- Handle missing values  
- Remove unwanted columns  
- Standardize datasets  

### 📊 Interactive Visualizations (Plotly)

- Histogram, Boxplot, Scatter, Line, Bar, Pie, Violin, KDE Density  
- Correlation Heatmap  
- Each chart includes **benefits & usage guidance**  
- Snapshots automatically saved for inclusion in PDF reports  

### 🤖 Model Training

- Auto-detects **Classification** or **Regression** based on target column  
- Suggests best-fit models with explanations  
- Dataset split into Training & Testing (visualized)  
- Metrics reported: **Accuracy, F1-score, Recall, R²**  
- Models saved as `.pickle` files for reusability  

### 📤 Export Reports

- Downloadable **ZIP report** containing:
  - 📝 Cleaned dataset (`.csv`)  
  - 🤖 Trained models (`.pickle`)  
  - 📈 Metrics (`.json`)  
  - 📑 Professional PDF report (includes cleaning summary, charts, training results)  

☠ DataForge by Mysterious | Hassan

## 📂 Project Structure

project/
│
├── modules/
│   ├── visualization.py       # Plotly visualizations
│   ├── model\_training.py      # ML model training logic
│   ├── report\_export.py       # Export & PDF generation
│
├── ui/
│   ├── data\_cleaning\_ui.py    # UI for cleaning
│   ├── visualization\_ui.py    # UI for charts
│   ├── model\_ui.py            # UI for training
│   ├── export\_ui.py           # UI for exporting
│
├── reports/                   # Generated reports & exports
├── models/                    # Trained models (.pickle)
├── requirements.txt           # Dependencies
└── app.py                     # Streamlit entry point

## ⚙️ Installation

1. **Clone the repository**

   git clone <https://github.com/Hassan-07-code/DataForge.git>
   cd dataforge

2. **Create virtual environment**

   python -m venv venv
   venv\Scripts\activate   # Windows

3. **Install dependencies**

   pip install -r requirements.txt

4. **Run the application**

   streamlit run app.py

## 📊 Workflow Overview

1. **Data Cleaning**
   Upload dataset → handle missing values → remove unwanted columns

2. **Visualization**
   Select charts → explore data → get automatic benefit insights

3. **Model Training**
   Choose target column → suggested model & reasoning → train & evaluate

4. **Export Report**
   Download ZIP with:

   - 📑 PDF (charts, metrics, cleaning summary)
   - 📝 Clean dataset (`.csv`)
   - 📈 Metrics (`.json`)
   - 🤖 Trained models (`.pickle`)

## 🛠️ Tech Stack

- **Streamlit** – UI framework
- **Plotly** – Interactive charts
- **scikit-learn** – ML models & evaluation
- **ReportLab** – PDF generation
- **Pandas** – Data wrangling
- **Kaleido** – Chart export

## ❤️ Credits

Developed with dedication by
**☠ Mysterious | Hassan**
 Muhammad Hassan Imtiaz
 AI Student || Flutter || Analysis Intelligence
