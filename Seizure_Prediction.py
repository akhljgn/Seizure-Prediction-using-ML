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
# 3. CUSTOM STYLING
# =====================================================================================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #b2dfdb 100%);
    }
    
    /* Card styling */
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    /* Section headers */
    .section-header {
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        color: #34495e;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #26a69a;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #26a69a, #00897b);
        color: white;
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
        background: linear-gradient(135deg, #00897b, #00695c);
        box-shadow: 0 6px 20px rgba(38, 166, 154, 0.4);
        transform: translateY(-2px);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: #e0f2f1;
        border: 2px dashed #26a69a;
        border-radius: 12px;
        padding: 1rem;
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
        background: white;
        border-radius: 20px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        margin: 2rem auto;
        max-width: 900px;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #26a69a, #00897b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    
    /* Result boxes */
    .result-box {
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
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
        color: #26a69a;
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
        st.error(f" Model files not found! Please ensure all required files are in the directory.")
        return None, None, None

# =====================================================================================
# 5. HOME PAGE
# =====================================================================================
def show_home_page():
    st.markdown('<h1 class="main-title"> EEG Seizure Prediction System</h1>', unsafe_allow_html=True)
    
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
    
    # st.markdown("<br>", unsafe_allow_html=True)
    
    # # Information cards
    # st.markdown("""
    # <div style="text-align: center; max-width: 800px; margin: 0 auto;">
    #     <h3 style="color: #2c3e50; margin-bottom: 1.5rem;">How It Works</h3>
    #     <p style="color: #7f8c8d; font-size: 1.1rem; line-height: 1.8; margin-bottom: 2rem;">
    #         Upload your EEG data file (CSV format with 178 features), and our system will:
    #         <br><br>
    #         1️⃣ Apply advanced <strong>noise filtering</strong> using Butterworth band-pass filters
    #         <br>
    #         2️⃣ Perform <strong>signal standardization</strong> and dimensionality reduction
    #         <br>
    #         3️⃣ Use <strong>machine learning</strong> to detect seizure patterns
    #         <br>
    #         4️⃣ Provide detailed <strong>analysis and visualization</strong> of results
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # # Key Features
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     st.markdown("""
    #     <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center; height: 100%;">
    #         <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
    #         <div style="font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Medical Grade</div>
    #         <div style="color: #7f8c8d; font-size: 1rem;">
    #             Clinically validated algorithms for accurate detection
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # with col2:
    #     st.markdown("""
    #     <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center; height: 100%;">
    #         <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
    #         <div style="font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Fast Analysis</div>
    #         <div style="color: #7f8c8d; font-size: 1rem;">
    #             Real-time processing with instant results
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # with col3:
    #     st.markdown("""
    #     <div style="background: white; padding: 2rem; border-radius: 16px; box-shadow: 0 8px 25px rgba(0,0,0,0.1); text-align: center; height: 100%;">
    #         <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
    #         <div style="font-size: 1.3rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem;">Detailed Reports</div>
    #         <div style="color: #7f8c8d; font-size: 1rem;">
    #             Comprehensive visualizations and metrics
    #         </div>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # st.markdown("<br><br>", unsafe_allow_html=True)
    
    # # Get Started Section
    # st.markdown("""
    # <div style="text-align: center; max-width: 600px; margin: 0 auto;">
    #     <h3 style="color: #2c3e50; margin-bottom: 1.5rem;">Ready to Analyze EEG Data?</h3>
    #     <p style="color: #7f8c8d; font-size: 1.1rem; margin-bottom: 2rem;">
    #         Click the button below to access the analysis dashboard and start detecting seizure activity.
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)
    
    # Center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(" Start Analysis", use_container_width=True):
            st.session_state.page = 'analysis'
            st.rerun()
    
    # Footer
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #95a5a6; font-size: 0.9rem; padding: 2rem;">
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
    
    st.markdown('<h1 class="main-title"> EEG Seizure Detection System</h1>', unsafe_allow_html=True)
    
    # Load model
    model, scaler, pca = load_assets()
    
    if not all([model, scaler, pca]):
        st.error(" Failed to load required model files. Please check your installation.")
        st.stop()
    
    # Layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    # ================== Left Column: Upload EEG Data ================== #
    with col1:
        st.markdown('<div class="section-header">Upload EEG Data</div>', unsafe_allow_html=True)
        
        # st.markdown("""
        # <div style="background: #e0f2f1; padding: 1rem; border-radius: 12px; margin-bottom: 1.5rem;">
        #     <p style="margin: 0; color: #00695c; font-weight: 500;">
        #         📋 <strong>File Requirements:</strong>
        #     </p>
        #     <ul style="color: #00897b; margin-top: 0.5rem; margin-bottom: 0;">
        #         <li>CSV format with 178 feature columns</li>
        #         <li>Each row represents one EEG sample</li>
        #         <li>Optional: Include 'y' column for validation</li>
        #     </ul>
        # </div>
        # """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose your EEG data file",
            type=['csv'],
            help="Upload a CSV file containing 178 EEG features per sample"
        )
        
        if uploaded_file is not None:
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            # Analyze button
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button(" Analyze EEG Data", use_container_width=True)
        else:
            analyze_button = False
    
    # ================== Right Column: Results Panel ================== #
    with col2:
        st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None and analyze_button:
            with st.spinner(" Analyzing EEG signals... Please wait."):
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
                            <h2 style="margin: 0;"> SEIZURE ACTIVITY DETECTED</h2>
                            <p style="font-size: 1.3rem; margin-top: 1rem; margin-bottom: 0;">
                                {seizure_count} out of {total} samples show seizure patterns
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box no-seizure">
                            <h2 style="margin: 0;"> NO SEIZURE DETECTED</h2>
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
                    st.error(f" Error: {str(e)}")
                    st.stop()
            
            # ================== Detailed Analysis Section ================== #
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
            
            # Visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=['No Seizure', 'Seizure'],
                    values=[no_seizure_count, seizure_count],
                    hole=0.4,
                    marker_colors=['#4facfe', '#f5576c']
                )])
                fig_pie.update_layout(
                    title="Prediction Distribution",
                    height=350
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # with viz_col2:
            #     confidence_scores = probabilities.max(axis=1)
            #     fig_hist = px.histogram(
            #         confidence_scores,
            #         nbins=30,
            #         labels={'value': 'Confidence Score'},
            #         title='Confidence Distribution'
            #     )
            #     fig_hist.update_layout(height=350)
            #     st.plotly_chart(fig_hist, use_container_width=True)
            
            # Performance metrics if labels provided
            # if has_labels:
            #     st.markdown("<br>", unsafe_allow_html=True)
            #     st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
                
            #     accuracy = accuracy_score(y_true, predictions)
            #     st.metric("Accuracy", f"{accuracy:.2%}")
                
            #     perf_col1, perf_col2 = st.columns(2)
                
                # with perf_col1:
                #     cm = confusion_matrix(y_true, predictions)
                #     fig_cm = px.imshow(
                #         cm,
                #         text_auto=True,
                #         labels=dict(x="Predicted", y="Actual"),
                #         x=['No Seizure', 'Seizure'],
                #         y=['No Seizure', 'Seizure'],
                #         title="Confusion Matrix",
                #         color_continuous_scale='Teal'
                #     )
                #     st.plotly_chart(fig_cm, use_container_width=True)
                
                # with perf_col2:
                #     report_dict = classification_report(y_true, predictions, output_dict=True)
                #     report_df = pd.DataFrame(report_dict).transpose()
                #     st.dataframe(
                #         report_df.style.format("{:.3f}"),
                #         use_container_width=True
                #     )
            
            # Download results
            st.markdown("<br>", unsafe_allow_html=True)
            results_df = df_raw.copy()
            results_df['Prediction'] = predictions
            results_df['Prediction_Label'] = results_df['Prediction'].map({0: 'No Seizure', 1: 'Seizure'})
            results_df['Seizure_Probability'] = probabilities[:, 1]
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Download Results",
                data=csv,
                file_name=f"eeg_analysis_{uploaded_file.name}",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.markdown(
                '<div style="text-align: center; color: #95a5a6; font-style: italic; margin-top: 100px; font-size: 1.2rem;">'
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