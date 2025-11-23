import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt

# =====================================================================================
# 1. PAGE CONFIGURATION
# =====================================================================================
st.set_page_config(
    page_title="EEG Seizure Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================================================
# 2. SESSION STATE INITIALIZATION
# =====================================================================================
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# =====================================================================================
# 3. CUSTOM STYLING WITH DARK MODE SUPPORT
# =====================================================================================
st.markdown("""
<style>
    /* ========== BROWSER THEME DETECTION ========== */
    :root {
        --primary-color: #26a69a;
        --primary-dark: #00897b;
        --primary-darker: #00695c;
    }
    
    /* Light mode (default browser preference) */
    @media (prefers-color-scheme: light) {
        :root {
            --text-primary: #2c3e50;
            --text-secondary: #7f8c8d;
            --bg-card: #ffffff;
            --bg-gradient: linear-gradient(135deg, #e8f5e9 0%, #b2dfdb 100%);
            --shadow: rgba(0, 0, 0, 0.1);
            --border: #e0e0e0;
            --info-bg: #e0f2f1;
            --info-border: #26a69a;
        }
    }
    
    /* Dark mode (browser preference) */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #e8eaed;
            --text-secondary: #9aa0a6;
            --bg-card: #1e1e1e;
            --bg-gradient: linear-gradient(135deg, #0a1929 0%, #1a2332 100%);
            --shadow: rgba(0, 0, 0, 0.4);
            --border: #333333;
            --info-bg: #1a3a3a;
            --info-border: #26a69a;
        }
    }
    
    /* Background gradient that adapts */
    .stApp {
        background: var(--bg-gradient);
    }
    
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px var(--shadow);
    }
    
    /* Card styling */
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: var(--bg-card);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px var(--shadow);
    }
    
    /* Section headers */
    .section-header {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-color);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white !important;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 16px;
        font-weight: 600;
        border: none;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(38, 166, 154, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-dark), var(--primary-darker));
        box-shadow: 0 6px 20px rgba(38, 166, 154, 0.4);
        transform: translateY(-2px);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: var(--info-bg);
        border: 2px dashed var(--info-border);
        border-radius: 12px;
        padding: 1rem;
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    button[kind="header"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* Home screen styling */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--bg-card);
        border-radius: 20px;
        box-shadow: 0 15px 40px var(--shadow);
        margin: 2rem auto;
        max-width: 900px;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Result boxes */
    .result-box {
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 25px var(--shadow);
    }
    
    .seizure {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .no-seizure {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: var(--primary-color) !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    /* Info boxes */
    .info-box {
        background: var(--info-bg);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 4px solid var(--primary-color);
    }
    
    .info-box p {
        margin: 0;
        color: var(--text-primary);
        font-weight: 500;
    }
    
    .info-box ul {
        color: var(--text-secondary);
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Spinner color */
    .stSpinner > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Download button specific */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark)) !important;
        color: white !important;
    }
    
    /* Alert/Error messages */
    .stAlert {
        background: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }
    
    /* Success message */
    .stSuccess {
        background-color: rgba(38, 166, 154, 0.1) !important;
        color: var(--text-primary) !important;
    }
    
    /* Placeholder text */
    .placeholder-text {
        text-align: center;
        color: var(--text-secondary);
        font-style: italic;
        margin-top: 100px;
        font-size: 1.2rem;
    }
    
    /* Footer text */
    .footer-text {
        text-align: center;
        color: var(--text-secondary);
        font-size: 0.9rem;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================================
# 4. HELPER FUNCTIONS
# =====================================================================================
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Applies a Butterworth band-pass filter to the EEG data."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=1)
    return y

@st.cache_resource
def load_assets():
    """Loads the ML model, scaler, and PCA object from disk."""
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        return model, scaler, pca
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure all required files are in the directory.")
        return None, None, None

# =====================================================================================
# 5. HOME PAGE
# =====================================================================================
def show_home_page():
    st.markdown('<h1 class="main-title">EEG Seizure Prediction System</h1>', unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h2 class="hero-title">Welcome</h2>
        <p class="hero-subtitle">
            An advanced AI-powered system that analyzes EEG signals to detect seizure activity 
            using <strong>Random Forest Classification</strong> with sophisticated signal processing techniques.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Analysis", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()
    
    # Footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer-text">
        <p>Powered by <strong>Random Forest</strong> with <strong>Advanced Signal Processing</strong></p>
        <p style="margin-top: 0.5rem;">EEG Analysis | Machine Learning | Healthcare AI</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================================
# 6. ANALYSIS PAGE
# =====================================================================================
def show_analysis_page():
    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown('<h1 class="main-title">EEG Seizure Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, pca = load_assets()
    
    if not all([model, scaler, pca]):
        st.error("Failed to load required model files. Please check your installation.")
        st.stop()
    
    # Layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    # ================== Left Column: Upload EEG Data ================== #
    with col1:
        st.markdown('<div class="section-header">Upload EEG Data</div>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose your EEG data file",
            type=['csv'],
            help="Upload a CSV file containing 178 EEG features per sample"
        )
        
        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully.")
            
            # Analyze button
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("Analyze EEG Data", use_container_width=True)
        else:
            analyze_button = False
    
    # ================== Right Column: Results Panel ================== #
    with col2:
        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None and analyze_button:
            with st.spinner("Analyzing EEG signals... Please wait."):
                try:
                    # Load data
                    df_raw = pd.read_csv(uploaded_file)
                    
                    # Check for labels
                    has_labels = 'y' in df_raw.columns
                    if has_labels:
                        X_raw = df_raw.drop('y', axis=1)
                        y_true = df_raw['y']
                    else:
                        X_raw = df_raw
                        y_true = None
                    
                    # Validate input
                    if X_raw.shape[1] != 178:
                        st.error(f"Expected 178 features, but found {X_raw.shape[1]} columns.")
                        st.stop()
                    
                    # Preprocessing pipeline
                    X_filtered = butter_bandpass_filter(X_raw.values, lowcut=0.5, highcut=50, fs=178)
                    X_scaled = scaler.transform(X_filtered)
                    X_pca = pca.transform(X_scaled)
                    
                    # Prediction
                    predictions = model.predict(X_pca)
                    probabilities = model.predict_proba(X_pca)
                    
                    # Calculate metrics
                    seizure_count = np.sum(predictions == 1)
                    no_seizure_count = np.sum(predictions == 0)
                    total = len(predictions)
                    
                    # Display main result
                    if seizure_count > 0:
                        st.markdown(f"""
                        <div class="result-box seizure">
                            <h2 style="margin: 0;">SEIZURE ACTIVITY DETECTED</h2>
                            <p style="font-size: 1.3rem; margin-top: 1rem; margin-bottom: 0;">
                                {seizure_count} out of {total} samples show seizure patterns
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box no-seizure">
                            <h2 style="margin: 0;">NO SEIZURE DETECTED</h2>
                            <p style="font-size: 1.3rem; margin-top: 1rem; margin-bottom: 0;">
                                All {total} samples appear normal
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    st.markdown("<br>", unsafe_allow_html=True)
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric("Total Samples", total)
                    metric_col2.metric("Seizure", seizure_count)
                    metric_col3.metric("Normal", no_seizure_count)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.stop()
            
            # ================== Detailed Analysis Section ================== #
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
            
            # Bar chart - full width
            fig_bar = go.Figure(data=[
                go.Bar(name='Count', x=['No Seizure', 'Seizure'], 
                       y=[no_seizure_count, seizure_count],
                       marker_color=['#4facfe', '#f5576c'],
                       text=[no_seizure_count, seizure_count],
                       textposition='auto')
            ])
            fig_bar.update_layout(
                title="Prediction Counts",
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e8eaed'),
                xaxis_title="Prediction Category",
                yaxis_title="Number of Samples"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Download results
            st.markdown("<br>", unsafe_allow_html=True)
            results_df = df_raw.copy()
            results_df['Prediction'] = predictions
            results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'No Seizure', 1: 'Seizure'})
            results_df['Seizure_Probability'] = probabilities[:, 1]
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results",
                data=csv,
                file_name=f"eeg_analysis_{uploaded_file.name}",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.markdown(
                '<div class="placeholder-text">'
                'Upload a file and click "Analyze" to see results'
                '</div>',
                unsafe_allow_html=True
            )

# =====================================================================================
# 7. PAGE ROUTING
# =====================================================================================
if st.session_state.page == 'home':
    show_home_page()
else:
    show_analysis_page()