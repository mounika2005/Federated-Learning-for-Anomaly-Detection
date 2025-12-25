


#  FLARE: Federated Learning–Based Explainable Intrusion Detection System

FLARE is a privacy-preserving, federated, and explainable Intrusion Detection System (IDS) designed to detect zero-day and unknown cyber attacks in distributed environments such as IoT, edge, and enterprise networks. The system combines Federated Learning, Deep Autoencoder–based anomaly detection, and Explainable AI (SHAP) to deliver accurate, transparent, and real-time intrusion detection without sharing raw network traffic data.

##  Key Features
- Zero-day and unknown attack detection  
- Privacy-preserving federated learning (no raw data sharing)  
- Deep autoencoder–based anomaly detection  
- Explainable AI (SHAP) for feature-level explanations  
- Real-time monitoring dashboard  
- Scalable for IoT, edge, and cloud environments  

##  Technologies Used
- Language: Python 3.8+  
- Deep Learning: PyTorch  
- Federated Learning: Flower (FLWR)  
- Explainable AI: SHAP  
- Data Processing: NumPy, Pandas, Scikit-learn  
- Dashboard: Streamlit, Plotly  
- Dataset: CICIDS2017  

##  Dataset
FLARE uses the CICIDS2017 dataset, which contains real-world benign and malicious network traffic suitable for evaluating intrusion detection systems.

##  Setup Instructions
### Prerequisites
- Python 3.8 or higher  
- pip package manager  
- Optional: CUDA-enabled GPU  

### Clone Repository
```bash
git clone https://github.com/your-username/flare.git
cd flare


##  Setup Instructions

### 1. Prerequisites

- Python 3.8+
- [Optional] CUDA-enabled GPU (PyTorch will use it if available)

### 2. Clone & Environment

```bash
# Clone the repository
git clone https://github.com/your-username/flare.git
cd flare

# Create Virtual Environment
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn flwr shap streamlit plotly matplotlib requests
```

### 4. Download Dataset

We use the **CICIDS2017** dataset. Run the helper script to download and extract it automatically:

```powershell
$env:PYTHONPATH='.'; python download_data.py
```

_This will create a `data/traffic.csv` file._

---

## 🏃‍♂️ How to Run

FLARE consists of 3 components that must run simultaneously. Open **3 separate terminal windows**.

### Terminal 1: The Server (Aggregator)

The server coordinates the Federated Learning rounds.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python src/server/server.py
```

_Wait until you see "Flower server running"._

### Terminal 2: The Client (Edge Device)

The client loads local data, trains the model, and performs anomaly detection.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python src/client/app.py
```

_The client will connect to the server, download the global model, train on local data, and upload updated weights._

### Terminal 3: The Dashboard (Control Center)

The dashboard visualizes real-time traffic, anomalies, and explanations.

```powershell
$env:PYTHONPATH='c:\proj\flare'; python -m streamlit run dashboard/dashboard.py
```

_Open the URL shown (usually `http://localhost:8501`)._

---

##  Interactive Demo Features

In the Dashboard:

1.  **Simulate Normal Traffic**: Click the Green button. The chart should be low/stable (Green dots).
2.  **Simulate Web Attack**: Click the Red button. The chart will spike (Red crosses) and Alerts will trigger.
3.  **Explainability**: Scroll down to see exactly _why_ an attack was detected (e.g., "High Flow Duration" or "Excessive SYN Flags").

---

##  Project Structure

```
flare/
├── src/
│   ├── client/
│   │   ├── model.py        # Deep Autoencoder Architecture (PyTorch)
│   │   ├── pipeline.py     # Inference & Thresholding Logic
│   │   ├── app.py          # Flower Client & Differential Privacy
│   │   └── explain.py      # SHAP Explainability Wrapper
│   ├── server/
│   │   └── server.py       # Flower Server Strategy
│   └── utils/
│       ├── data_loader.py  # CICIDS2017 Data Parser
│       └── preprocessing.py# MinMax Scaler & Data Loaders
├── dashboard/
│   └── dashboard.py        # Streamlit + Plotly UI
├── data/                   # Dataset directory
└── download_data.py        # Dataset downloader script
```
=======
# Federated-Learning-for-Anomaly-Detection
>>>>>>> 4fcd5a8cdf4da235aaeff9c8ceebf0ac2267d0c2
