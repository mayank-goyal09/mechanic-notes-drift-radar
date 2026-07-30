<div align="center">

# ⚡ Drift Radar — Industrial Equipment Failure & Text Analysis Hub

**Deployed App:** [drift-radar.app](https://mechanic-notes-drift-radar-project.streamlit.app/)

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Outfit&weight=700&size=32&duration=3500&pause=1000&color=38BDF8&center=true&vCenter=true&width=900&height=50&lines=Industrial+Equipment+Failure+Analysis+🏭;Density-Based+NLP+Log+Clustering+🧠;Real-Time+Data+Drift+Detection+🚨)](https://git.io/typing-svg)

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Clustering](https://img.shields.io/badge/Clustering-HDBSCAN-10b981?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Keywords](https://img.shields.io/badge/Keywords-YAKE-00f2fe?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

<br/>

[![🚀 Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-Drift_Radar_Dashboard-38bdf8?style=for-the-badge&labelColor=0f172a)](https://mechanic-notes-drift-radar-project.streamlit.app/)
[![GitHub Stars](https://img.shields.io/github/stars/mayank-goyal09/mechanic-notes-drift-radar?style=for-the-badge&color=ffd700)](https://github.com/mayank-goyal09/mechanic-notes-drift-radar/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/mayank-goyal09/mechanic-notes-drift-radar?style=for-the-badge&color=87ceeb)](https://github.com/mayank-goyal09/mechanic-notes-drift-radar/network)

<br/>

![Drift Radar Banner](assets/drift_radar_banner.png)

<br/>

### 🧠 **Using HDBSCAN Density-Based Clustering to group mechanic logs** 

### **From Raw Maintenance Notes → Real-Time Failure & Drift Diagnosis** 🏭

</div>

---

## ⚡ **THE ANALYSIS AT A GLANCE**

<table>
<tr>
<td width="50%">

### 🎯 **What This Project Does**

**Drift Radar** is a machine learning dashboard designed to process unstructured mechanic repair notes, clean industrial jargon, automatically cluster failures using **HDBSCAN**, extract key concepts using **YAKE**, and flag newly emerging failure modes (data drift) in real-time.

**The ML Pipeline:**
- 🧹 **Text Preprocessing** → Expand shorthand (e.g. `hydr` -> `hydraulic`) and strip numeric noise
- 🔢 **Vectorization** → TF-IDF n-grams feature extraction
- 🧬 **Clustering** → HDBSCAN density-based clustering to automatically group similar failure modes without needing a predefined $K$ value
- 🏷️ **Auto-Labeling** → YAKE keyword extraction to identify core engineering themes of each cluster
- 🚨 **Drift Simulation** → Injects high-frequency anomalous logs to test the model's re-clustering detection

</td>
<td width="50%">

### ✨ **Key Highlights**

| Feature | Details |
|---------|---------|
| ⚙️ **Clustering Control** | Real-time HDBSCAN parameter tuning (Min Cluster Size, Min Samples) |
| 📊 **Failure Topology** | 2D PCA cluster scatter maps and frequency plots |
| 🚨 **Drift Simulator** | Inject and detect new failure categories on-the-fly |
| 🧠 **Real-Time Diagnosis** | Classify new mechanic notes with prediction confidence |
| 🎨 **UI Design** | Dark theme, glassmorphism, and neon accents |
| 💾 **Caching** | Cached data generation & model resource fit pipelines |
| 📉 **Noise Handling** | Excludes unrelated notes as outliers/noise |

</td>
</tr>
</table>

---

## 🛠️ **TECHNOLOGY STACK**

<div align="center">

![Tech Stack](https://skillicons.dev/icons?i=python,sklearn,plotly,streamlit)

</div>

| **Category** | **Technologies** | **Purpose** |
|:------------:|:-----------------|:------------|
| 🐍 **Core Language** | Python 3.13+ | Primary development language |
| 🧬 **Clustering Engine** | HDBSCAN | Outlier-resilient density clustering |
| 🏷️ **Keywords** | YAKE | Automated keyword and metadata extraction |
| 🎨 **Frontend** | Streamlit | Interactive, customized web application dashboard |
| 📈 **Visualization** | Plotly | Interactive PCA scatter plots, gauges, and bar charts |
| ⚡ **NLP & Vectors** | Scikit-Learn | TF-IDF Vectorizer and PCA dimensionality reduction |
| 📂 **Data Handling** | Pandas / NumPy | Data structuring, filtering, and synthetic simulation |

---

## 🔬 **HOW DRIFT RADAR WORKS**

```mermaid
graph LR
    A[🏭 Mechanic Notes] --> B[🧹 Text Preprocessing]
    B --> C[🔢 TF-IDF Vectorization]
    C --> D[🧬 HDBSCAN Clustering]
    D --> E[🏷️ YAKE Keyword Labeling]
    E --> F[🚨 Drift & Anomaly Detector]
    F --> G[📊 Drift Radar Dashboard]
    
    style A fill:#38bdf8,color:#000
    style D fill:#818cf8,color:#fff
    style F fill:#f59e0b,color:#000
    style G fill:#10b981,color:#fff
```

### **The Pipeline Breakdown:**

<table>
<tr>
<td>

#### 🧹 **1. Preprocessing & Clean**
Removes machine serial codes and punctuation, converts text to lowercase, and maps mechanic shorthand abbreviations (like `hydr` -> `hydraulic`, `repl` -> `replace`, `vibr` -> `vibration`, `noisy` -> `noise`).

</td>
<td>

#### 🔢 **2. Feature Vectorization**
Extracts word and bi-gram patterns using `TfidfVectorizer` to represent each log numerically based on semantic importance.

</td>
</tr>
<tr>
<td>

#### 🧬 **3. Density-Based Clustering**
Runs `HDBSCAN` on the dense TF-IDF vectors, automatically isolating stable clusters of failure categories while filtering out outliers as uncategorized noise.

</td>
<td>

#### 📊 **4. Auto-Labeling & Diagnostics**
Extracts keywords from clustered logs via `YAKE` to dynamically label failure modes. Predicts clusters for new logs and detects emerging concepts (drift) when anomalous clusters form.

</td>
</tr>
</table>

---

## 📂 **PROJECT STRUCTURE**

```
⚡ Drift Radar/
│
├── 📊 app.py                        # Main Streamlit application, page config, custom CSS & UI tabs
├── 🧠 main.ipynb                    # Jupyter Notebook for experimental model testing & EDA
│
├── 📁 assets/
│   └── 🖼️ drift_radar_banner.png   # Project banner image
│
├── 📁 data/
│   ├── 📄 raw_repair_logs.csv       # Pre-split/original synthetic repair logs
│   └── 📄 processed_repair_logs.csv # Cleaned repair logs with pipeline cached output
│
├── 📦 requirements.txt             # Project dependencies
└── 📖 README.md                     # You are here! 🎉
```

---

## 🚀 **QUICK START GUIDE**

### **Step 1: Clone the Repository** 📥

```bash
git clone https://github.com/mayank-goyal09/mechanic-notes-drift-radar.git
cd mechanic-notes-drift-radar
```

### **Step 2: Create Virtual Environment** 🐍

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux
```

### **Step 3: Install Dependencies** 📦

```bash
pip install -r requirements.txt
```

### **Step 4: Launch Drift Radar** ⚡

```bash
streamlit run app.py
```

---

## 📚 **SKILLS DEMONSTRATED**

| **Category** | **Skills** |
|:-------------|:-----------|
| 🧠 **Unstructured Text NLP** | Regular expressions cleanups, jargon mapping, TF-IDF n-gram embeddings |
| 🔮 **Density-Based Clustering** | Hyperparameter tuning (Min Cluster Size, Min Samples) with HDBSCAN |
| 📊 **Dimensionality Reduction** | Visualizing high-dimensional TF-IDF vectors using Principal Component Analysis (PCA) |
| 🏷️ **Keyword Extraction** | Statistical unsupervised metadata labeling using YAKE |
| 🎨 **UI/UX Customization** | Advanced glassmorphism Streamlit layouts, pulsing status elements, and custom CSS |
| 🚨 **ML Drift Simulation** | Simulating high-frequency data drifts and dynamically generating re-clustering plots |

---

## 🔮 **FUTURE ENHANCEMENTS**

- [ ] 🤖 **LLM Maintenance Assistant**: Integrating local Ollama models to summarize repair tasks and suggest solutions.
- [ ] 🔌 **API Integration**: Creating REST API endpoints to receive logs from ERP software (e.g., SAP PM).
- [ ] 📈 **Time Series Forecasting**: Predicting when failure modes are likely to spike based on maintenance dates.
- [ ] 🗺️ **Interactive UMAP Map**: Replacing PCA with UMAP for richer high-dimensional structural representations.

---

## 👨‍💻 **CONNECT WITH ME**

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-mayank--goyal09-181717?style=for-the-badge&logo=github)](https://github.com/mayank-goyal09)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mayank_Goyal-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/mayank-goyal-4b8756363/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit_Site-38bdf8?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mayank-goyal09.github.io/)

**Mayank Goyal**  
📊 Data Analyst | 🧠 NLP Enthusiast | 🏭 Predictive Maintenance Specialist

</div>

---

<div align="center">

### ⚡ **Built with AI & ❤️ by Mayank Goyal**

*"Uncovering failure modes, preventing operational drift."* ⚙️🧠

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:38bdf8,100:10b981&height=120&section=footer)

</div>
