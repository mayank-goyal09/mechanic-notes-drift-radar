import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import hdbscan
import yake
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# 1. Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Machine Error Detection Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Custom CSS (Dark Theme, Glassmorphism, Neon Accents, Modern Typography)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Main Background and Text */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', 'Inter', sans-serif !important;
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Navigation Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Custom Card Design */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: rgba(96, 165, 250, 0.2);
        transform: translateY(-2px);
    }
    
    /* Metric styling */
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    
    /* Real-time diagnostics styling */
    .alert-card-normal {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
    }
    
    .alert-card-noise {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.1);
    }
    
    .alert-card-drift {
        background: rgba(245, 158, 11, 0.12);
        border: 2px dashed rgba(245, 158, 11, 0.5);
        border-radius: 12px;
        padding: 24px;
        margin-top: 15px;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.1);
    }
    
    /* Custom Headers */
    h1, h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    /* Style all top navigation buttons */
    button[aria-label*="Dashboard Overview"],
    button[aria-label*="Repair Log Inspector"],
    button[aria-label*="Real-time Diagnosis"],
    button[aria-label*="Drift Simulation"] {
        border-radius: 12px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        height: 52px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* Inactive button style (Secondary) */
    button[aria-label*="Dashboard Overview"][data-testid="stBaseButton-secondary"],
    button[aria-label*="Repair Log Inspector"][data-testid="stBaseButton-secondary"],
    button[aria-label*="Real-time Diagnosis"][data-testid="stBaseButton-secondary"],
    button[aria-label*="Drift Simulation"][data-testid="stBaseButton-secondary"] {
        background-color: rgba(30, 41, 59, 0.4) !important;
        color: #94a3b8 !important;
    }

    /* Inactive Hover state */
    button[aria-label*="Dashboard Overview"][data-testid="stBaseButton-secondary"]:hover,
    button[aria-label*="Repair Log Inspector"][data-testid="stBaseButton-secondary"]:hover,
    button[aria-label*="Real-time Diagnosis"][data-testid="stBaseButton-secondary"]:hover,
    button[aria-label*="Drift Simulation"][data-testid="stBaseButton-secondary"]:hover {
        background: rgba(56, 189, 248, 0.1) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
        color: #38bdf8 !important;
        box-shadow: 0 4px 20px rgba(56, 189, 248, 0.2) !important;
        transform: translateY(-2px);
    }

    /* Active button style (Primary) */
    button[aria-label*="Dashboard Overview"][data-testid="stBaseButton-primary"],
    button[aria-label*="Repair Log Inspector"][data-testid="stBaseButton-primary"],
    button[aria-label*="Real-time Diagnosis"][data-testid="stBaseButton-primary"],
    button[aria-label*="Drift Simulation"][data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.2), rgba(129, 140, 248, 0.2)) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.35), inset 0 0 8px rgba(56, 189, 248, 0.1) !important;
        font-weight: 700 !important;
    }

    /* Active Hover state */
    button[aria-label*="Dashboard Overview"][data-testid="stBaseButton-primary"]:hover,
    button[aria-label*="Repair Log Inspector"][data-testid="stBaseButton-primary"]:hover,
    button[aria-label*="Real-time Diagnosis"][data-testid="stBaseButton-primary"]:hover,
    button[aria-label*="Drift Simulation"][data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.25), rgba(129, 140, 248, 0.25)) !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.45) !important;
        transform: translateY(-2px);
    }

    /* Sidebar Header */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 15px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .pulsing-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }

    .status-text {
        font-size: 0.85rem;
        font-weight: 600;
        color: #10b981;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Cards */
    .sidebar-card {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        margin-bottom: 15px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
    }

    .sidebar-card-title {
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        margin-bottom: 12px !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding-bottom: 6px !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }

    .spec-row {
        display: flex !important;
        justify-content: space-between !important;
        font-size: 0.8rem !important;
        padding: 4px 0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.02) !important;
    }

    .spec-row:last-child {
        border-bottom: none !important;
    }

    .spec-label {
        color: #94a3b8 !important;
    }

    .spec-value {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Text Preprocessing & Cleaning Utility
# -------------------------------------------------------------
def clean_mechanic_note(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove machine/serial numbers
    # Expand jargon
    jargon_map = {"hydr": "hydraulic", "repl": "replace", "vibr": "vibration", "noisy": "noise"}
    for short, full in jargon_map.items():
        text = text.replace(short, full)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

# -------------------------------------------------------------
# 3. Data Loading & Generation Pipeline
# -------------------------------------------------------------
@st.cache_data
def load_or_generate_data():
    csv_path = 'data/processed_repair_logs.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Parse dates
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    # If not exists, generate synthetic data
    templates = {
        "Hydraulic": ["hydr leak on joint", "piston pressure low", "seal repl in hydr system"],
        "Electrical": ["blown fuse in panel B", "wiring frayed on motor", "short circuit detected"],
        "Mechanical": ["vibr in main drive", "bearing noisy", "gearbox metal shavings found"],
        "Noise": ["routine inspection", "cleaned workstation", "discussed with shift lead"]
    }
    
    data = []
    np.random.seed(42)
    start_date = datetime(2026, 7, 1)
    for i in range(500):
        date = start_date + timedelta(days=np.random.randint(0, 28))
        category = np.random.choice(list(templates.keys()), p=[0.2, 0.2, 0.2, 0.4])
        note = np.random.choice(templates[category])
        data.append({"timestamp": date, "note": note, "actual_cat": category})
        
    df = pd.DataFrame(data).sort_values("timestamp")
    df['clean_note'] = df['note'].apply(clean_mechanic_note)
    
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/raw_repair_logs.csv', index=False)
    df.to_csv(csv_path, index=False)
    return df

# -------------------------------------------------------------
# 4. Model Caching & Vectorization
# -------------------------------------------------------------
@st.cache_resource
def fit_models(df, min_cluster_size=10, min_samples=2):
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    X = vectorizer.fit_transform(df['clean_note'])
    
    # HDBSCAN Clusterer
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    # Fit HDBSCAN on dense array
    cluster_labels = clusterer.fit_predict(X.toarray())
    
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X.toarray())
    
    return vectorizer, clusterer, cluster_labels, pca, pca_coords

# Load base data
df_base = load_or_generate_data()

# -------------------------------------------------------------
# 5. Sidebar Model Tuning & Specifications
# -------------------------------------------------------------
with st.sidebar:
    # Pulsing Green Dot + Online Status
    st.markdown("""
    <div class="sidebar-header">
        <span class="pulsing-dot"></span>
        <span class="status-text">Engine Online</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Engine Diagnostics")
    st.write("Tune model clustering hyperparameters below to re-cluster failure logs in real-time.")
    
    min_cluster_size = st.slider(
        "Min Cluster Size", 
        min_value=2, 
        max_value=50, 
        value=10, 
        step=1, 
        help="The minimum size of a group to be considered a cluster."
    )
    min_samples = st.slider(
        "Min Samples", 
        min_value=1, 
        max_value=10, 
        value=2, 
        step=1, 
        help="The number of samples in a neighborhood for a point to be considered a core point."
    )
    
    # Model Properties & System Specs
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">🤖 Model Properties</div>
        <div class="spec-row"><span class="spec-label">Vectorizer</span><span class="spec-value">TF-IDF (1,2)</span></div>
        <div class="spec-row"><span class="spec-label">Metric</span><span class="spec-value">Euclidean</span></div>
        <div class="spec-row"><span class="spec-label">Selection</span><span class="spec-value">EOM</span></div>
    </div>
    <div class="sidebar-card">
        <div class="sidebar-card-title">🖥️ System Specs</div>
        <div class="spec-row"><span class="spec-label">Python</span><span class="spec-value">3.13</span></div>
        <div class="spec-row"><span class="spec-label">Scikit-Learn</span><span class="spec-value">1.8.0</span></div>
        <div class="spec-row"><span class="spec-label">Streamlit</span><span class="spec-value">1.49.1</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Clear Cache button
    if st.button("Reset Resource Cache", use_container_width=True):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

# Run model pipeline with slider values
vectorizer, clusterer, cluster_labels, pca, pca_coords = fit_models(df_base, min_cluster_size, min_samples)

# Create copy of dataframe and assign cluster labels
df_clustered = df_base.copy()
df_clustered['cluster_id'] = cluster_labels

# Calculate baseline metrics globally
n_logs = len(df_clustered)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

# -------------------------------------------------------------
# YAKE Keyword Extractor & Summarizer
# -------------------------------------------------------------
@st.cache_resource
def get_keyword_extractor():
    return yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=3)

kw_extractor = get_keyword_extractor()

@st.cache_data
def get_cluster_metadata(df_data, unique_cluster_ids):
    meta = {}
    for cid in unique_cluster_ids:
        if cid == -1:
            meta[cid] = ["Uncategorized", "Miscellaneous Notes"]
        else:
            combined_text = " ".join(df_data[df_data['cluster_id'] == cid]['clean_note'])
            keywords = kw_extractor.extract_keywords(combined_text)
            meta[cid] = [kw[0] for kw in keywords]
    return meta

unique_clusters = sorted(df_clustered['cluster_id'].unique())
cluster_meta = get_cluster_metadata(df_clustered, unique_clusters)

# -------------------------------------------------------------
# 6. Streamlit Layout & Navigation
# -------------------------------------------------------------
# Initialize session state for app mode
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "📊 Dashboard Overview"

# Header area with Title, Subtitle, and Model Configuration
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.markdown("<div class='main-title'>⚡ Drift Radar</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #94a3b8; margin-top: -5px; margin-bottom: 20px; font-size: 1.15rem; font-weight: 500;'>Industrial Equipment Failure & Text Analysis Hub</div>", unsafe_allow_html=True)
with h_col2:
    st.markdown("""
    <div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 12px 16px; border-radius: 12px; text-align: right; box-shadow: 0 4px 15px rgba(0,0,0,0.15); backdrop-filter: blur(5px);'>
        <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600;'>Active Engine</div>
        <div style='font-size: 0.95rem; font-weight: 700; color: #38bdf8; margin-top: 2px;'>HDBSCAN + YAKE</div>
    </div>
    """, unsafe_allow_html=True)

# Top Navigation Bar (Horizontal Columns)
nav_cols = st.columns(4)
nav_options = [
    "📊 Dashboard Overview",
    "🔍 Repair Log Inspector",
    "🧠 Real-time Diagnosis",
    "🚨 Drift Simulation"
]

for i, option in enumerate(nav_options):
    with nav_cols[i]:
        is_active = (st.session_state.app_mode == option)
        btn_type = "primary" if is_active else "secondary"
        if st.button(option, key=f"nav_btn_{i}", use_container_width=True, type=btn_type):
            st.session_state.app_mode = option
            st.rerun()

# Divider
st.markdown("<div style='margin: 15px 0 25px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.05);'></div>", unsafe_allow_html=True)

# Read active mode from session state
app_mode = st.session_state.app_mode

# -------------------------------------------------------------
# Tab 1: Dashboard Overview
# -------------------------------------------------------------
if app_mode == "📊 Dashboard Overview":
    st.markdown("### 📊 Fleet Diagnostic Overview")
    
    # Layout with metrics
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Total Repair Logs</div>
            <div class="metric-value">{n_logs}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Detected Failure Modes</div>
            <div class="metric-value">{n_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Uncategorized / Noise</div>
            <div class="metric-value">{n_noise}</div>
        </div>
        """, unsafe_allow_html=True)
        
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### Failure Modes Frequency")
        # Prepare bar chart data
        cluster_counts = df_clustered['cluster_id'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster ID', 'Log Count']
        cluster_counts['Description'] = cluster_counts['Cluster ID'].apply(
            lambda x: f"Mode {x}: " + ", ".join(cluster_meta.get(x, ["Unknown"]))
        )
        # Sort values
        cluster_counts = cluster_counts.sort_values(by='Cluster ID')
        
        fig_bar = px.bar(
            cluster_counts,
            y='Description',
            x='Log Count',
            orientation='h',
            color='Log Count',
            color_continuous_scale='Bluered',
            template='plotly_dark'
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with c2:
        st.markdown("#### Failure Topology (2D PCA Cluster Map)")
        
        pca_df = pd.DataFrame(pca_coords, columns=['PCA 1', 'PCA 2'])
        pca_df['Cluster ID'] = df_clustered['cluster_id'].astype(str)
        pca_df['Note'] = df_clustered['note']
        
        fig_scatter = px.scatter(
            pca_df,
            x='PCA 1',
            y='PCA 2',
            color='Cluster ID',
            hover_data=['Note'],
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            template='plotly_dark'
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.markdown("#### Latest Maintenance Logs")
    latest_df = df_clustered.sort_values(by='timestamp', ascending=False).head(10)[['timestamp', 'note', 'clean_note', 'cluster_id']]
    latest_df['Keywords'] = latest_df['cluster_id'].apply(lambda x: ", ".join(cluster_meta.get(x, [])))
    st.dataframe(latest_df, use_container_width=True)

# -------------------------------------------------------------
# Tab 2: Repair Log Inspector
# -------------------------------------------------------------
elif app_mode == "🔍 Repair Log Inspector":
    st.markdown("### 🔍 Filter and Query Maintenance Logs")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # Cluster filter
        selected_cluster = st.selectbox(
            "Filter by Cluster ID",
            options=["All"] + unique_clusters,
            index=0
        )
    with col2:
        # Date selection
        min_date = df_clustered['timestamp'].min().date()
        max_date = df_clustered['timestamp'].max().date()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    with col3:
        # Text search query
        search_query = st.text_input("Search notes by keyword", placeholder="e.g. bearing, fuse, leak")
        
    # Filtering process
    filtered_df = df_clustered.copy()
    
    if selected_cluster != "All":
        filtered_df = filtered_df[filtered_df['cluster_id'] == selected_cluster]
        
    if len(date_range) == 2:
        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        filtered_df = filtered_df[(filtered_df['timestamp'] >= start_dt) & (filtered_df['timestamp'] < end_dt)]
        
    if search_query:
        filtered_df = filtered_df[
            filtered_df['note'].str.contains(search_query, case=False, na=False) |
            filtered_df['clean_note'].str.contains(search_query, case=False, na=False)
        ]
        
    st.markdown(f"**Found {len(filtered_df)} matches**")
    
    # Map keywords for display
    display_df = filtered_df.copy()
    display_df['Keywords'] = display_df['cluster_id'].apply(lambda x: ", ".join(cluster_meta.get(x, [])))
    
    st.dataframe(
        display_df[['timestamp', 'note', 'clean_note', 'cluster_id', 'Keywords']],
        use_container_width=True
    )
    
    # Download option
    csv_data = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered CSV",
        data=csv_data,
        file_name="filtered_repair_logs.csv",
        mime="text/csv"
    )

# -------------------------------------------------------------
# Tab 3: Real-time Diagnosis (Inference)
# -------------------------------------------------------------
elif app_mode == "🧠 Real-time Diagnosis":
    st.markdown("### 🧠 Predictive Failure Model Diagnostic")
    st.write("Enter a new maintenance note below to run the preprocessing, vectorization, and cluster classification.")
    
    input_note = st.text_area("Mechanic Note input", placeholder="e.g., Frayed electrical panel B cable, blown fuse repl", height=100)
    
    if st.button("Diagnose Issue"):
        if input_note.strip() == "":
            st.error("Please enter a valid mechanic note.")
        else:
            # 1. Clean note
            cleaned = clean_mechanic_note(input_note)
            
            # 2. Vectorize
            vectorized_sample = vectorizer.transform([cleaned])
            
            # 3. Approximate predict with HDBSCAN
            label, strength = hdbscan.approximate_predict(clusterer, vectorized_sample.toarray())
            predicted_id = int(label[0])
            confidence = float(strength[0])
            
            # 4. Extract local YAKE keywords
            local_kws = [kw[0] for kw in kw_extractor.extract_keywords(cleaned)]
            
            st.markdown("#### Diagnostic Report")
            
            if predicted_id == -1:
                st.markdown(f"""
                <div class="alert-card-noise">
                    <h3 style="margin-top: 0; color: #f87171 !important;">⚠️ Uncategorized Log (Noise)</h3>
                    <p><b>Original Note:</b> "{input_note}"</p>
                    <p><b>Cleaned Note:</b> "{cleaned}"</p>
                    <p><b>Prediction Confidence:</b> {confidence:.2%}</p>
                    <p><b>Extracted Note Keywords:</b> {", ".join(local_kws) if local_kws else "None"}</p>
                    <p style="margin-bottom: 0;"><i>This log does not match any existing failure modes. It might be a routine note or a unique incident.</i></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                mode_kws = cluster_meta.get(predicted_id, [])
                st.markdown(f"""
                <div class="alert-card-normal">
                    <h3 style="margin-top: 0; color: #34d399 !important;">✅ Classified Failure Mode {predicted_id}</h3>
                    <p><b>Original Note:</b> "{input_note}"</p>
                    <p><b>Cleaned Note:</b> "{cleaned}"</p>
                    <p><b>HDBSCAN Map Confidence:</b> {confidence:.2%}</p>
                    <p><b>Associated Mode Keywords:</b> {", ".join(mode_kws)}</p>
                    <p style="margin-bottom: 0; color: #10b981;"><b>Diagnostic:</b> Consistent with historical patterns for this category.</p>
                </div>
                """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Tab 4: Drift Simulation
# -------------------------------------------------------------
elif app_mode == "🚨 Drift Simulation":
    st.markdown("### 🚨 Emerging Failure Mode Detector")
    st.write("In industrial operations, new equipment failure patterns emerge over time. This section simulates drift by injecting new anomalous logs and re-clustering to automatically identify newly forming clusters.")
    
    st.info("Baseline model currently has **%d** distinct failure mode clusters." % n_clusters)
    
    # Drift injection simulation
    if "drift_simulated" not in st.session_state:
        st.session_state.drift_simulated = False
        
    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        if st.button("Inject Anomalies"):
            st.session_state.drift_simulated = True
    with c_btn2:
        if st.button("Reset Simulator"):
            st.session_state.drift_simulated = False
            
    if st.session_state.drift_simulated:
        st.warning("⚠️ Simulating data drift: 15 logs containing 'sensor overheating', 'thermal detector high', and 'unit temperature high' anomalies have been added to the pipeline.")
        
        # Build simulated drift logs (15 identical logs to form a dense new cluster)
        more_drift = [
            {
                "timestamp": datetime(2026, 7, 28, 10, 0) + timedelta(minutes=int(i*20)),
                "note": "overheating sensor high",
                "actual_cat": "Drift"
            }
            for i in range(15)
        ]
        
        df_drift = pd.concat([df_base, pd.DataFrame(more_drift)], ignore_index=True)
        df_drift['clean_note'] = df_drift['note'].apply(clean_mechanic_note)
        
        # Fit vectorizer and clusterer on combined data
        X_drift = vectorizer.transform(df_drift['clean_note'])
        
        # Run HDBSCAN
        drift_labels = clusterer.fit_predict(X_drift.toarray())
        df_drift['cluster_id'] = drift_labels
        
        # Identify new clusters
        old_ids = set(cluster_labels)
        new_ids = set(drift_labels)
        detected_drift_ids = new_ids - old_ids - {-1}
        
        if detected_drift_ids:
            for d_id in detected_drift_ids:
                # Get keywords for this specific new cluster
                drift_cluster_text = " ".join(df_drift[df_drift['cluster_id'] == d_id]['clean_note'])
                drift_kws = [kw[0] for kw in kw_extractor.extract_keywords(drift_cluster_text)]
                
                st.markdown(f"""
                <div class="alert-card-drift">
                    <h3 style="margin-top: 0; color: #f59e0b !important;">🚨 DRIFT DETECTED: New Failure Mode Formed!</h3>
                    <p><b>Newly Formed Cluster ID:</b> {d_id}</p>
                    <p><b>Extracted Core Keywords:</b> {", ".join(drift_kws)}</p>
                    <p style="margin-bottom: 0;"><b>Recommended Engineering Action:</b> Flag these entries and assign a new failure category code in the database matching: <i>{", ".join(drift_kws)}</i></p>
                </div>
                """, unsafe_allow_html=True)
                
            # Plot new PCA representation
            pca_coords_drift = pca.transform(X_drift.toarray())
            pca_df_drift = pd.DataFrame(pca_coords_drift, columns=['PCA 1', 'PCA 2'])
            pca_df_drift['Cluster ID'] = df_drift['cluster_id'].astype(str)
            # Label the drift cluster specifically for visuals
            pca_df_drift['Cluster ID'] = pca_df_drift['Cluster ID'].apply(
                lambda x: f"NEW Mode {x} (DRIFT)" if int(x) in detected_drift_ids else f"Mode {x}"
            )
            pca_df_drift['Note'] = df_drift['note']
            
            fig_scatter_drift = px.scatter(
                pca_df_drift,
                x='PCA 1',
                y='PCA 2',
                color='Cluster ID',
                hover_data=['Note'],
                color_discrete_sequence=px.colors.qualitative.Alphabet,
                template='plotly_dark'
            )
            fig_scatter_drift.update_layout(
                title="PCA Cluster Map (With Emerging Failure Mode)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=450,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_scatter_drift, use_container_width=True)
        else:
            st.info("Re-clustering completed. The injected logs were absorbed as noise or existing clusters. No new distinct failure modes detected.")
    else:
        st.info("Simulator idle. Click the 'Inject Anomalies' button above to simulate incoming anomalous repair data.")
