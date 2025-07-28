# 🛡️ AI-Based Cyber Threat Detection System

An advanced AI-powered platform to detect potential cyber threats in real time using Machine Learning. The system leverages network traffic analysis and pattern recognition techniques to identify malicious behavior and reduce response time by 47%.

---

## 🚀 Key Features

- 🔍 **Real-Time Cyber Threat Detection** using AI models
- 🧠 **High Detection Accuracy (92%)** using Isolation Forest and Decision Tree
- 🌐 **Interactive Dashboard** to monitor threats live
- 🕵️‍♂️ **Network Traffic Analysis** with Wireshark logs
- 📈 **Risk Categorization** with severity indicators
- 🔐 Built-in **Threat Classification Engine**

---

## 🛠️ Tech Stack

| Layer        | Technologies Used                                 |
|--------------|---------------------------------------------------|
| 💻 Frontend  | React.js, Chart.js, TailwindCSS                   |
| 🔙 Backend   | Python, Flask                                     |
| 🧠 ML Models | Isolation Forest, Decision Tree Classifier        |
| 🐍 Libraries | Scikit-learn, Pandas, NumPy                       |
| 🗂️ Dataset   | Custom + Wireshark captured traffic               |
| 🗃️ Database  | MongoDB                                           |

---

## 📂 Folder Structure

├── client/ # React.js frontend
│ ├── components/ # Reusable UI elements
│ └── App.js # Main frontend logic
├── server/ # Flask backend
│ ├── models/ # ML models and preprocessing scripts
│ └── app.py # API endpoints for predictions
├── data/ # Network traffic samples and logs
├── utils/ # Feature extraction and security functions
└── README.md


---

## 🧠 Detection Models

- **Isolation Forest**: Detects anomalies by isolating traffic patterns that differ from normal behavior.
- **Decision Tree**: Provides interpretable classification of threats based on extracted network features.

These models were trained on labeled datasets (including packet logs and known malicious patterns), achieving a **92% detection rate** and significantly reducing **false positives**.

---

## 📊 Dashboard Insights

- 🚨 Displays threat alerts in real-time  
- 📈 Shows confidence score of detection  
- 📊 Visualizes threat types (DDoS, Port Scans, Malware, etc.)  
- 🧠 Offers risk level based on model outputs

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/mitsy145/AI-CyberThreat-Detection.git
cd AI-CyberThreat-Detection

# 2. Start the backend
cd server
pip install -r requirements.txt
python app.py

# 3. Start the frontend
cd ../client
npm install
npm start
