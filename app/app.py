import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import sys
import os

# Add src directory to path (handle both running from root and app directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from preprocessing import DataPreprocessor
    from train_models import ModelTrainer
    from evaluation import ModelEvaluator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
try:
    st.set_page_config(
        page_title="Liver Cirrhosis Disease Prediction",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    # Page config can only be called once; ignore if already set
    pass

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #28a745;
        text-align: center;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #ffc107;
        text-align: center;
        margin: 1rem 0;
    }
    .danger-card {
        background-color: #f8d7da;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #dc3545;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    /* Top navigation styling */
    .main-nav {
        display: flex;
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    .main-nav div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
    }
    .main-nav div[role="radiogroup"] > label {
        padding: 0.35rem 1.2rem;
        border-radius: 999px;
        border: 1px solid #1f77b4;
        background-color: #ffffff;
        color: #1f77b4;
        font-weight: 500;
        cursor: pointer;
    }
    .main-nav div[role="radiogroup"] > label[data-selected="true"],
    .main-nav div[role="radiogroup"] > label[aria-checked="true"] {
        background-color: #1f77b4;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Liver Cirrhosis Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6c757d;">Prediction of Liver Cirrhosis Disease using XGBoost Majority Voting Technique</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 0.9rem; color: #6c757d; font-style: italic;">Implementation based on IEEE ICMLAS-2025 Paper Methodology</p>', unsafe_allow_html=True)

# Top navigation (Home / About)
with st.container():
    st.markdown('<div class="main-nav">', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Home", "About"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None

# Pretrained assets (load before pages render)
MODELS_DIR = os.path.join(project_root, 'models')
RESULTS_FILE = os.path.join(MODELS_DIR, 'all_results.pkl')
PREPROCESSOR_FILE = os.path.join(MODELS_DIR, 'preprocessor.pkl')


@st.cache_resource(show_spinner=False)
def _load_pretrained_assets():
    """
    Load pretrained models + preprocessing metadata if available.
    Expected files:
    - models/all_results.pkl
    - models/preprocessor.pkl  (contains scaler + feature_columns)
    """
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(PREPROCESSOR_FILE):
        return None
    try:
        results = joblib.load(RESULTS_FILE)
        prep = joblib.load(PREPROCESSOR_FILE)
        return {
            "results": results,
            "scaler": prep.get("scaler"),
            "feature_columns": prep.get("feature_columns"),
        }
    except Exception:
        return None


def _bootstrap_session_with_pretrained():
    """Populate session_state from pretrained artifacts if present."""
    if st.session_state.models_trained:
        return
    assets = _load_pretrained_assets()
    if not assets:
        return

    # Validate required pieces
    if not assets.get("results") or assets.get("scaler") is None or not assets.get("feature_columns"):
        return

    st.session_state.results = assets["results"]
    st.session_state.scaler = assets["scaler"]
    st.session_state.feature_columns = assets["feature_columns"]

    # Create evaluation_results for evaluation/dashboard pages
    evaluator = ModelEvaluator()
    evaluator.set_results(st.session_state.results)
    evaluator.calculate_metrics()
    st.session_state.evaluation_results = {
        "metrics_df": evaluator.metrics_df,
        "results": evaluator.results,
    }

    st.session_state.models_trained = True


_bootstrap_session_with_pretrained()

# Main content
# Home Page
if page == "Home":
    st.markdown('<h2 class="sub-header">Liver Cirrhosis Disease Prediction</h2>', unsafe_allow_html=True)
    st.caption("Proposed XGBoost Majority Voting Model (Internal Ensemble Technique) + Stage Classification (Ascites, Hepatomegaly, Spiders, Edema).")

    if not st.session_state.models_trained:
        st.warning("Pre-trained model is not loaded. Go to **About** to train once (or include the `models/` folder when sharing the project).")

    with st.form("prediction_form"):
        st.markdown('<h3 class="sub-header">Patient Parameters (IEEE)</h3>', unsafe_allow_html=True)

        st.markdown('<h4 class="sub-header">Demographics</h4>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=50)
        with col2:
            sex = st.selectbox("Sex", ["Female", "Male"])
        with col3:
            drug = st.selectbox("Drug", ["No", "Yes"])

        st.markdown('<h4 class="sub-header">Disease Stage Indicators</h4>', unsafe_allow_html=True)
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            ascites = st.selectbox("Ascites", ["No", "Yes"])
        with col5:
            hepatomegaly = st.selectbox("Hepatomegaly", ["No", "Yes"])
        with col6:
            spiders = st.selectbox("Spiders", ["No", "Yes"])
        with col7:
            edema = st.selectbox("Edema", ["No", "Yes"])

        st.markdown('<h4 class="sub-header">Biochemical Features (Ranges)</h4>', unsafe_allow_html=True)
        st.caption("Ranges include: Bilirubin, Triglycerides, Platelets, Cholesterol, Albumin, Copper, Alkaline Phosphatase, SGOT.")
        col8, col9 = st.columns(2)
        with col8:
            bilirubin = st.number_input("Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
            triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=0.0, max_value=500.0, value=120.0, step=0.1)
            platelets = st.number_input("Platelets (10^9/L)", min_value=0, max_value=1000, value=300)
            cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=400.0, value=180.0, step=0.1)
        with col9:
            albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, value=4.0, step=0.1)
            copper = st.number_input("Copper (mcg/dL)", min_value=0.0, max_value=300.0, value=100.0, step=0.1)
            alkaline_phosphatase = st.number_input("Alkaline Phosphatase (U/L)", min_value=0, max_value=2000, value=100)
            sgot = st.number_input("SGOT / AST (U/L)", min_value=0, max_value=1000, value=30)

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        if not st.session_state.models_trained or 'results' not in st.session_state or 'XGBoost_Majority_Voting' not in st.session_state.results:
            st.error("Model not loaded. Go to **About** and train models once.")
        elif st.session_state.scaler is None or not st.session_state.feature_columns:
            st.error("Preprocessor metadata missing. Go to **About** and retrain once.")
        else:
            input_data_model = {
                'Age': age,
                'Sex': 1 if sex == 'Male' else 0,
                'Drug': 1 if drug == 'Yes' else 0,
                'Ascites': 1 if ascites == 'Yes' else 0,
                'Hepatomegaly': 1 if hepatomegaly == 'Yes' else 0,
                'Spiders': 1 if spiders == 'Yes' else 0,
                'Edema': 1 if edema == 'Yes' else 0,
                'Bilirubin': bilirubin,
                'Triglycerides': triglycerides,
                'Platelets': platelets,
                'Cholesterol': cholesterol,
                'Albumin': albumin,
                'Copper': copper,
                'Alkaline_Phosphatase': alkaline_phosphatase,
                'SGOT': sgot
            }

            stage_data = {
                'Ascites': input_data_model['Ascites'],
                'Hepatomegaly': input_data_model['Hepatomegaly'],
                'Spiders': input_data_model['Spiders'],
                'Edema': input_data_model['Edema']
            }

            model = st.session_state.results['XGBoost_Majority_Voting']['model']
            scaler = st.session_state.scaler
            feature_columns = st.session_state.feature_columns

            X_input = pd.DataFrame([input_data_model])[feature_columns]
            X_scaled = scaler.transform(X_input)
            prediction = int(model.predict(X_scaled)[0])
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(proba[1] if prediction == 1 else proba[0])

            st.markdown('<h3 class="sub-header">Prediction Summary</h3>', unsafe_allow_html=True)
            col_res, col_stage = st.columns(2)
            with col_res:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="danger-card">
                        <h2>Liver Cirrhosis Detected</h2>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Risk Level:</strong> High</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>No Cirrhosis Detected</h2>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p><strong>Risk Level:</strong> Low</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<h3 class="sub-header">Disease Stage Classification</h3>', unsafe_allow_html=True)
            indicators_df = pd.DataFrame({
                'Indicator': ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema'],
                'Status': [
                    'Present' if stage_data['Ascites'] == 1 else 'Absent',
                    'Present' if stage_data['Hepatomegaly'] == 1 else 'Absent',
                    'Present' if stage_data['Spiders'] == 1 else 'Absent',
                    'Present' if stage_data['Edema'] == 1 else 'Absent'
                ],
                'Value': [
                    stage_data['Ascites'],
                    stage_data['Hepatomegaly'],
                    stage_data['Spiders'],
                    stage_data['Edema']
                ]
            })
            st.dataframe(indicators_df, use_container_width=True, hide_index=True)

            stage_score = stage_data['Ascites'] + stage_data['Hepatomegaly'] + stage_data['Spiders'] + stage_data['Edema']
            if prediction == 0:
                stage_label = 'No Disease'
            else:
                if stage_score <= 1:
                    stage_label = 'Early Stage'
                elif stage_score == 2:
                    stage_label = 'Moderate Stage'
                else:
                    stage_label = 'Advanced Stage'

            st.markdown(f"""
            <div class="metric-card">
                <h3>Predicted Stage: {stage_label}</h3>
                <p><strong>Stage Score:</strong> {stage_score} out of 4 indicators</p>
                <p><em>Ascites, Hepatomegaly, Spiders, Edema (per IEEE methodology)</em></p>
            </div>
            """, unsafe_allow_html=True)

# Data Analysis Page
elif page == "Data Analysis":
    st.markdown('<h2 class="sub-header">Data Analysis & Exploration</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Liver Cirrhosis Dataset", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data_loaded = True
            st.session_state.df = df
            
            st.success(f"Dataset loaded successfully! Shape: {df.shape}")
            
            # Basic information
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", f"{df.shape[0]:,}")
            with col2:
                st.metric("Features", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Data Types", len(df.dtypes.unique()))
            
            # Dataset preview
            st.markdown('<h3 class="sub-header">Dataset Preview</h3>', unsafe_allow_html=True)
            st.dataframe(df.head(10))
            
            # Data information
            st.markdown('<h3 class="sub-header">Dataset Information</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Types:**")
                st.write(df.dtypes)
            
            with col2:
                st.markdown("**Missing Values:**")
                missing_data = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': (missing_data.values / len(df)) * 100
                })
                st.dataframe(missing_df[missing_df['Missing Count'] > 0])
            
            # Statistical summary
            st.markdown('<h3 class="sub-header">Statistical Summary</h3>', unsafe_allow_html=True)
            st.dataframe(df.describe())
            
            # Visualizations
            st.markdown('<h3 class="sub-header">Data Visualizations</h3>', unsafe_allow_html=True)
            
            # Target distribution
            if 'target' in df.columns or df.columns[-1] in ['0', '1', 'disease', 'label']:
                target_col = 'target' if 'target' in df.columns else df.columns[-1]
                
                fig = px.pie(
                    values=df[target_col].value_counts().values,
                    names=['No Disease', 'Disease'],
                    title="Target Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Correlation heatmap for numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap"
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    
    else:
        st.info("Please upload a CSV file to begin data analysis.")
        
        # Sample data information
        st.markdown('<h3 class="sub-header">Expected Dataset Format</h3>', unsafe_allow_html=True)
        st.markdown("""
        The dataset should align with IEEE input parameters:
        - **Age, Sex, Drug, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin**
        - **Ranges of:** Triglycerides, Platelets, Cholesterol, Albumin, Copper, Alkaline Phosphatase, SGOT (and Bilirubin)
        - **Target**: 0 = Liver Cirrhosis absent, 1 = Liver Cirrhosis present
        - Optional: Remove irrelevant columns (e.g. untitled); missing values handled by preprocessing.
        """)

# Model Training Page
elif page == "Model Training":
    st.markdown('<h2 class="sub-header">Model Training & Ensemble Creation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("Please load a dataset first in the Data Analysis page.")
    else:
        if st.button("🚀 Start Model Training", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Initialize components
                    preprocessor = DataPreprocessor()
                    trainer = ModelTrainer()
                    evaluator = ModelEvaluator()
                    
                    # Preprocess data - Use all IEEE paper features
                    st.info("Preprocessing data...")
                    
                    # Check if dataset file exists
                    dataset_path = os.path.join(project_root, 'data', 'liver_cirrhosis.csv')
                    if not os.path.exists(dataset_path):
                        st.error(f"Dataset file not found at: {dataset_path}")
                        st.info("Please ensure the dataset file exists in the 'data' folder, or upload it in the 'Data Analysis' page first.")
                        st.stop()
                    
                    try:
                        processed_data = preprocessor.preprocess_pipeline(dataset_path, 'target', use_all_features=True)
                    except Exception as prep_error:
                        st.error(f"Error during preprocessing: {prep_error}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()
                    
                    # Train models
                    st.info("Training individual models...")
                    results = trainer.train_complete_pipeline(
                        processed_data['X_train'],
                        processed_data['y_train'],
                        processed_data['X_test'],
                        processed_data['y_test']
                    )
                    
                    # Evaluate models
                    st.info("Evaluating models...")
                    
                    # Validate results before evaluation
                    if not results or len(results) == 0:
                        st.error("No model results to evaluate. Training may have failed.")
                        st.stop()
                    
                    evaluator.set_results(results)
                    
                    # Debug: Show what results we have (can be removed in production)
                    with st.expander("🔍 Debug Information (Click to view)"):
                        st.write(f"**Found {len(results)} model results**")
                        st.write(f"**Model names:** {list(results.keys())}")
                        if results:
                            sample_key = list(results.keys())[0]
                            st.write(f"**Sample result structure for '{sample_key}':**")
                            st.json(results[sample_key] if isinstance(results[sample_key], dict) else str(results[sample_key]))
                    
                    # Verify evaluator has results
                    if not hasattr(evaluator, 'results') or not evaluator.results:
                        st.error("Evaluator results not set properly.")
                        st.stop()
                    
                    # Ensure evaluator has results before calling evaluate_all
                    if not hasattr(evaluator, 'results') or not evaluator.results:
                        st.error("Evaluator results not set. Attempting to set them again...")
                        evaluator.set_results(results)

                    # Manually calculate metrics first to ensure it works
                    try:
                        evaluator.calculate_metrics()
                        if evaluator.metrics_df is None or evaluator.metrics_df.empty:
                            st.error("Metrics calculation failed - metrics_df is empty")
                            st.info(f"Results available: {list(evaluator.results.keys()) if evaluator.results else 'None'}")
                            st.stop()
                    except Exception as calc_error:
                        st.error(f"Error calculating metrics: {calc_error}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()

                    evaluation_results = evaluator.evaluate_all(
                        processed_data['y_test'],
                        save_plots=False
                    )

                    # Ensure metrics_df is in the results
                    if 'metrics_df' not in evaluation_results:
                        st.warning("metrics_df not in evaluation_results, using evaluator.metrics_df")
                        evaluation_results['metrics_df'] = evaluator.metrics_df

                    # Debug: Check what was returned (in expander)
                    if evaluation_results:
                        with st.expander("🔍 Evaluation Debug Info (Click to view)"):
                            st.write(f"**Evaluation keys:** {list(evaluation_results.keys())}")
                            if 'metrics_df' in evaluation_results:
                                st.write(f"**Metrics DF shape:** {evaluation_results['metrics_df'].shape}")
                                st.write(f"**Metrics DF columns:** {list(evaluation_results['metrics_df'].columns)}")
                                if not evaluation_results['metrics_df'].empty:
                                    st.write("**First few rows:**")
                                    st.dataframe(evaluation_results['metrics_df'].head())
                    
                    st.session_state.models_trained = True
                    st.session_state.results = results
                    st.session_state.evaluation_results = evaluation_results
                    st.session_state.scaler = processed_data['scaler']
                    st.session_state.feature_columns = processed_data['feature_columns']

                    # Persist preprocessing metadata so others can run with pretrained models
                    try:
                        os.makedirs(MODELS_DIR, exist_ok=True)
                        joblib.dump(
                            {
                                "scaler": processed_data["scaler"],
                                "feature_columns": processed_data["feature_columns"],
                            },
                            PREPROCESSOR_FILE,
                        )
                    except Exception as save_prep_error:
                        st.warning(f"Could not save preprocessor metadata: {save_prep_error}")
                    
                    st.success("Model training completed successfully!")
                    
                    # Display results summary
                    st.markdown('<h3 class="sub-header">Training Results Summary</h3>', unsafe_allow_html=True)
                    
                    # IEEE Methodology Note
                    st.info("""
                    **IEEE Methodology Note:** The Proposed XGBoost Majority Voting Technique is an **Internal Ensemble Method within XGBoost itself**, 
                    not a VotingClassifier across different algorithms. It uses multiple XGBoost models with different random seeds, where:
                    - Initial predictions are generated using the average of target values
                    - Residuals are calculated for each iteration
                    - Decision trees are built using gain and similarity scores
                    - Residuals are repeatedly recalculated over N iterations
                    - Majority voting is applied internally within XGBoost
                    - Final classification is determined by majority occurrence (greater than N/2)
                    """)
                    
                    # Check if evaluation_results has metrics_df
                    if not evaluation_results or 'metrics_df' not in evaluation_results:
                        st.error("Evaluation results are missing or incomplete.")
                        st.info(f"Available keys: {list(evaluation_results.keys()) if evaluation_results else 'None'}")
                        # Try to create metrics_df manually from results
                        if 'results' in evaluation_results:
                            st.info("Attempting to create metrics_df from results...")
                            try:
                                evaluator.calculate_metrics()
                                metrics_df = evaluator.metrics_df
                                evaluation_results['metrics_df'] = metrics_df
                            except Exception as calc_error:
                                st.error(f"Failed to calculate metrics: {calc_error}")
                                st.stop()
                        else:
                            st.stop()
                    else:
                        metrics_df = evaluation_results['metrics_df']
                    
                    # Additional validation
                    if metrics_df is None:
                        st.error("Metrics dataframe is None. Evaluation may have failed.")
                        st.info("Attempting to recalculate metrics...")
                        try:
                            evaluator.calculate_metrics()
                            metrics_df = evaluator.metrics_df
                            evaluation_results['metrics_df'] = metrics_df
                        except Exception as calc_error:
                            st.error(f"Failed to recalculate metrics: {calc_error}")
                            st.stop()
                    
                    # Validate metrics_df has required columns
                    if metrics_df.empty:
                        st.error("Metrics dataframe is empty. Training may have failed.")
                        st.info("Checking results structure...")
                        if hasattr(evaluator, 'results') and evaluator.results:
                            st.write(f"Results available: {list(evaluator.results.keys())}")
                            # Try to manually create metrics
                            try:
                                evaluator.calculate_metrics()
                                metrics_df = evaluator.metrics_df
                                evaluation_results['metrics_df'] = metrics_df
                                if metrics_df.empty:
                                    st.error("Still empty after recalculation. Please check training logs.")
                                    st.stop()
                            except Exception as calc_error:
                                st.error(f"Failed to recalculate: {calc_error}")
                                st.stop()
                        else:
                            st.stop()
                    
                    required_columns = ['Model', 'Accuracy', 'Recall', 'F1-Score']
                    missing_columns = [col for col in required_columns if col not in metrics_df.columns]
                    if missing_columns:
                        st.error(f"Metrics dataframe is missing required columns: {missing_columns}")
                        st.info(f"Available columns: {list(metrics_df.columns)}")
                        st.info("Attempting to fix...")
                        # Try to recalculate
                        try:
                            evaluator.calculate_metrics()
                            metrics_df = evaluator.metrics_df
                            evaluation_results['metrics_df'] = metrics_df
                            missing_columns = [col for col in required_columns if col not in metrics_df.columns]
                            if missing_columns:
                                st.error(f"Still missing columns after recalculation: {missing_columns}")
                                st.stop()
                        except Exception as calc_error:
                            st.error(f"Failed to fix: {calc_error}")
                            st.stop()
                    
                    # Separate into three sections: Traditional Models, XGBoost Individual, XGBoost Majority Voting
                    traditional_models = ['SVM', 'KNN', 'Naive_Bayes', 'AdaBoost']
                    xgboost_individual = ['XGBoost_Individual']
                    xgboost_majority_voting = ['XGBoost_Majority_Voting']
                    
                    # Create dataframes for three sections
                    traditional_df = metrics_df[metrics_df['Model'].isin(traditional_models)].copy()
                    xgboost_ind_df = metrics_df[metrics_df['Model'].isin(xgboost_individual)].copy()
                    xgboost_mv_df = metrics_df[metrics_df['Model'].isin(xgboost_majority_voting)].copy()
                    
                    # Section 1: Traditional Models
                    st.markdown('<h4 class="sub-header">📊 Traditional Machine Learning Models</h4>', unsafe_allow_html=True)
                    if not traditional_df.empty:
                        traditional_df_sorted = traditional_df.sort_values('Accuracy', ascending=False)
                        st.dataframe(traditional_df_sorted[['Model', 'Accuracy', 'Recall', 'F1-Score']].round(4), 
                                   use_container_width=True)
                        
                        # Show best traditional model
                        best_traditional = traditional_df_sorted.loc[traditional_df_sorted['Accuracy'].idxmax()]
                        st.info(f"**Best Traditional Model:** {best_traditional['Model']} - {best_traditional['Accuracy']:.4f} ({best_traditional['Accuracy']*100:.2f}%)")
                    else:
                        st.warning("No traditional models found in results.")
                    
                    st.markdown("---")
                    
                    # Section 2: XGBoost Individual Model
                    st.markdown('<h4 class="sub-header">🔬 XGBoost Individual Model</h4>', unsafe_allow_html=True)
                    if not xgboost_ind_df.empty:
                        xgboost_ind_sorted = xgboost_ind_df.sort_values('Accuracy', ascending=False)
                        st.dataframe(xgboost_ind_sorted[['Model', 'Accuracy', 'Recall', 'F1-Score']].round(4), 
                                   use_container_width=True)
                    else:
                        st.warning("XGBoost Individual model not found in results.")
                    
                    st.markdown("---")
                    
                    # Section 3: Proposed XGBoost Majority Voting Technique
                    st.markdown('<h4 class="sub-header">🚀 Proposed XGBoost Majority Voting Model (Internal Ensemble Technique)</h4>', unsafe_allow_html=True)
                    if not xgboost_mv_df.empty:
                        xgboost_mv_sorted = xgboost_mv_df.sort_values('Accuracy', ascending=False)
                        st.dataframe(xgboost_mv_sorted[['Model', 'Accuracy', 'Recall', 'F1-Score']].round(4), 
                                   use_container_width=True)
                        
                        # Show XGBoost Majority Voting result with IEEE reference
                        if 'XGBoost_Majority_Voting' in xgboost_mv_sorted['Model'].values:
                            voting_result = xgboost_mv_sorted[xgboost_mv_sorted['Model'] == 'XGBoost_Majority_Voting'].iloc[0]
                            st.success(f"**Proposed XGBoost Majority Voting:** {voting_result['Accuracy']:.4f} ({voting_result['Accuracy']*100:.2f}%)")
                            st.markdown(f"**IEEE Reported Accuracy:** 99.4% (Target Benchmark)")
                            st.caption("This is an Internal Ensemble Technique within XGBoost, not a VotingClassifier across different algorithms.")
                            
                            # Show if we're close to IEEE target
                            if voting_result['Accuracy'] >= 0.994:
                                st.success("Achieved IEEE target accuracy!")
                            elif voting_result['Accuracy'] >= 0.99:
                                st.warning(f"Close to IEEE target (within {((voting_result['Accuracy'] - 0.994) * 100):.2f}%)")
                    else:
                        st.warning("Proposed XGBoost Majority Voting model not found in results.")
                    
                    # Overall comparison chart
                    st.markdown('<h4 class="sub-header">📈 Complete Model Comparison</h4>', unsafe_allow_html=True)
                    
                    # Comprehensive validation before plotting
                    with st.expander("🔍 DataFrame Validation (Click to view)"):
                        st.write(f"**Metrics DF Type:** {type(metrics_df)}")
                        st.write(f"**Is None:** {metrics_df is None}")
                        if metrics_df is not None:
                            st.write(f"**Is Empty:** {metrics_df.empty}")
                            st.write(f"**Shape:** {metrics_df.shape}")
                            st.write(f"**Has 'columns' attr:** {hasattr(metrics_df, 'columns')}")
                            if hasattr(metrics_df, 'columns'):
                                st.write(f"**Columns:** {list(metrics_df.columns)}")
                            st.write(f"**First few rows:**")
                            if not metrics_df.empty:
                                st.dataframe(metrics_df.head())
                    
                    if metrics_df is None:
                        st.error("❌ Metrics dataframe is None. Cannot create chart.")
                        st.stop()
                    
                    if not hasattr(metrics_df, 'columns'):
                        st.error("❌ Metrics dataframe doesn't have 'columns' attribute.")
                        st.stop()
                    
                    if metrics_df.empty:
                        st.error("❌ Metrics dataframe is empty. Cannot create chart.")
                        st.info("Please check the debug information above to see what went wrong.")
                        st.stop()
                    
                    if 'Model' not in metrics_df.columns:
                        st.error(f"❌ Metrics dataframe missing 'Model' column.")
                        st.info(f"Available columns: {list(metrics_df.columns)}")
                        st.stop()
                    
                    if 'Accuracy' not in metrics_df.columns:
                        st.error(f"❌ Metrics dataframe missing 'Accuracy' column.")
                        st.info(f"Available columns: {list(metrics_df.columns)}")
                        st.stop()
                    
                    # Create performance comparison chart with color coding
                    try:
                        # Create a fresh copy with only required columns to avoid serialization issues
                        plot_data = metrics_df[['Model', 'Accuracy']].copy()
                        metrics_df_sorted = plot_data.sort_values('Accuracy', ascending=True)
                        
                        # Final validation before plotting
                        if metrics_df_sorted.empty:
                            st.error("❌ Sorted metrics dataframe is empty. Cannot create chart.")
                            st.stop()
                        
                        if 'Model' not in metrics_df_sorted.columns or 'Accuracy' not in metrics_df_sorted.columns:
                            st.error(f"❌ Sorted dataframe missing required columns.")
                            st.info(f"Available: {list(metrics_df_sorted.columns)}")
                            st.stop()
                        
                        # Color map: Traditional models in blue, XGBoost Individual in green, Proposed Voting in red
                        colors = []
                        for model in metrics_df_sorted['Model']:
                            if model in traditional_models:
                                colors.append('#4ECDC4')  # Blue for traditional models
                            elif model == 'XGBoost_Individual':
                                colors.append('#96CEB4')  # Light green for XGBoost Individual
                            elif model == 'XGBoost_Majority_Voting':
                                colors.append('#FF4757')  # Red for proposed voting (best)
                            else:
                                colors.append('#FFEAA7')  # Yellow for others
                        
                        # Convert to dict format to avoid serialization issues
                        plot_dict = {
                            'Model': metrics_df_sorted['Model'].tolist(),
                            'Accuracy': metrics_df_sorted['Accuracy'].tolist()
                        }
                        plot_df = pd.DataFrame(plot_dict)
                        
                        # Final check
                        if plot_df.empty or 'Model' not in plot_df.columns or 'Accuracy' not in plot_df.columns:
                            st.error("❌ Failed to create plot data structure.")
                            st.info(f"Plot DF shape: {plot_df.shape}, columns: {list(plot_df.columns)}")
                            st.stop()
                        
                        # Create the plot with explicit data validation
                        fig = px.bar(
                            plot_df,
                            x='Model',
                            y='Accuracy',
                            title='Model Accuracy Comparison: Traditional Models vs XGBoost Individual vs Proposed XGBoost Majority Voting',
                            color='Accuracy',
                            color_continuous_scale='Viridis',
                            text='Accuracy',
                        )
                        fig.update_traces(
                            texttemplate='%{text:.4f}',
                            textposition='outside',
                            marker_color=colors,
                        )
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            yaxis_title='Accuracy',
                            height=500,
                        )
                        # Add IEEE accuracy reference line
                        fig.add_hline(
                            y=0.994,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="IEEE Target: 99.4%",
                            annotation_position="right",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as plot_error:
                        st.error(f"Error creating comparison chart: {plot_error}")
                        st.info(f"Metrics DF info: shape={metrics_df.shape if metrics_df is not None else 'None'}, columns={list(metrics_df.columns) if metrics_df is not None and not metrics_df.empty else 'None'}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
                    
                    # Display complete metrics table
                    st.markdown('<h4 class="sub-header">📋 Complete Metrics Table</h4>', unsafe_allow_html=True)
                    display_columns = ['Model', 'Accuracy', 'Recall', 'F1-Score']
                    if 'AUC' in metrics_df.columns:
                        display_columns.append('AUC')
                    st.dataframe(metrics_df[display_columns].round(4), 
                               use_container_width=True)
                    
                    # Highlight the best model (Proposed XGBoost Majority Voting)
                    best_model = None
                    best_existing = None
                    improvement = 0
                    if not metrics_df.empty:
                        best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
                        traditional_df = metrics_df[metrics_df['Model'].isin(['SVM', 'KNN', 'Naive_Bayes', 'AdaBoost'])]
                        if not traditional_df.empty:
                            best_existing = traditional_df.loc[traditional_df['Accuracy'].idxmax()]
                            improvement = ((best_model['Accuracy'] - best_existing['Accuracy']) / best_existing['Accuracy']) * 100
                    
                    if best_model is not None:
                        improvement_text = ""
                        if best_existing is not None:
                            improvement_text = f"<p><strong>Improvement:</strong> {improvement:.2f}% better than best Traditional Model ({best_existing['Model']}: {best_existing['Accuracy']:.4f})</p>"
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Best Performing Model</h3>
                            <p><strong>{best_model['Model']}</strong> achieved the highest accuracy of <strong>{best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)</strong></p>
                            <p>Recall: {best_model['Recall']:.4f} | F1-Score: {best_model['F1-Score']:.4f}</p>
                            {improvement_text}
                            <p><strong>IEEE Reported Accuracy:</strong> 99.4% (Target Benchmark)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("Best model information will be displayed once training completes successfully.")
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    st.error("Please ensure you have uploaded a valid dataset.")
        
        if st.session_state.models_trained:
            st.success("Models have been trained successfully!")
            
            # Show model information with clear categorization
            st.markdown('<h3 class="sub-header">Trained Models Overview</h3>', unsafe_allow_html=True)
            
            if 'results' in st.session_state:
                results = st.session_state.results
                
                # Section 1: Traditional Models
                st.markdown('<h4 class="sub-header">📊 Traditional Machine Learning Models</h4>', unsafe_allow_html=True)
                traditional_models_list = ['SVM', 'KNN', 'Naive_Bayes', 'AdaBoost']
                
                cols = st.columns(4)
                for idx, model_name in enumerate(traditional_models_list):
                    if model_name in results:
                        with cols[idx % 4]:
                            result = results[model_name]
                            st.metric(
                                label=model_name,
                                value=f"{result['accuracy']*100:.2f}%",
                                delta=f"F1: {result['f1_score']:.4f}"
                            )
                
                st.markdown("---")
                
                # Section 2: XGBoost Individual Model
                st.markdown('<h4 class="sub-header">🔬 XGBoost Individual Model</h4>', unsafe_allow_html=True)
                
                if 'XGBoost_Individual' in results:
                    xgb_result = results['XGBoost_Individual']
                    st.markdown("**XGBoost (Individual):**")
                    st.metric(
                        label="Accuracy",
                        value=f"{xgb_result['accuracy']*100:.2f}%"
                    )
                    st.write(f"• Recall: {xgb_result['recall']:.4f}")
                    st.write(f"• F1-Score: {xgb_result['f1_score']:.4f}")
                
                st.markdown("---")
                
                # Section 3: Proposed XGBoost Majority Voting Model
                st.markdown('<h4 class="sub-header">🚀 Proposed XGBoost Majority Voting Model (Internal Ensemble Technique)</h4>', unsafe_allow_html=True)
                
                if 'XGBoost_Majority_Voting' in results:
                    voting_result = results['XGBoost_Majority_Voting']
                    xgb_acc = results.get('XGBoost_Individual', {}).get('accuracy', 0)
                    improvement = ((voting_result['accuracy'] - xgb_acc) * 100) if xgb_acc > 0 else 0
                    
                    st.markdown("**Proposed XGBoost Majority Voting (Internal Ensemble):**")
                    st.metric(
                        label="Accuracy",
                        value=f"{voting_result['accuracy']*100:.2f}%",
                        delta=f"+{improvement:.2f}% vs XGBoost Individual" if improvement > 0 else None
                    )
                    st.write(f"• Recall: {voting_result['recall']:.4f}")
                    st.write(f"• F1-Score: {voting_result['f1_score']:.4f}")
                    if voting_result.get('auc'):
                        st.write(f"• AUC: {voting_result['auc']:.4f}")
                    
                    st.markdown("**IEEE Reported Accuracy:** 99.4% (Target Benchmark)")
                    st.caption("This is an Internal Ensemble Technique within XGBoost, not a VotingClassifier across different algorithms.")
                    
                    # Show improvement over traditional models
                    if 'evaluation_results' in st.session_state:
                        metrics_df = st.session_state.evaluation_results['metrics_df']
                        traditional_models_list = ['SVM', 'KNN', 'Naive_Bayes', 'AdaBoost']
                        traditional_avg = metrics_df[metrics_df['Model'].isin(traditional_models_list)]['Accuracy'].mean()
                        improvement_traditional = ((voting_result['accuracy'] - traditional_avg) / traditional_avg) * 100
                        st.info(f"**Improvement over traditional models average: +{improvement_traditional:.2f}%**")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.markdown('<h2 class="sub-header">Model Performance Evaluation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first in the Model Training page.")
    else:
        st.success("Models are ready for evaluation!")
        
        # Performance metrics
        if 'evaluation_results' in st.session_state:
            metrics_df = st.session_state.evaluation_results['metrics_df']
            
            # Separate into three sections: Traditional Models, XGBoost Individual, XGBoost Majority Voting
            traditional_models = ['SVM', 'KNN', 'Naive_Bayes', 'AdaBoost']
            xgboost_individual = ['XGBoost_Individual']
            xgboost_majority_voting = ['XGBoost_Majority_Voting']
            
            # Summary Statistics
            st.markdown('<h3 class="sub-header">📊 Performance Summary</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            traditional_df = metrics_df[metrics_df['Model'].isin(traditional_models)]
            xgboost_ind_df = metrics_df[metrics_df['Model'].isin(xgboost_individual)]
            xgboost_mv_df = metrics_df[metrics_df['Model'].isin(xgboost_majority_voting)]
            
            with col1:
                st.metric("Traditional Models Avg", f"{traditional_df['Accuracy'].mean()*100:.2f}%")
            with col2:
                if not xgboost_ind_df.empty:
                    st.metric("XGBoost Individual", f"{xgboost_ind_df['Accuracy'].iloc[0]*100:.2f}%")
                else:
                    st.metric("XGBoost Individual", "N/A")
            with col3:
                if not xgboost_mv_df.empty:
                    voting_acc = xgboost_mv_df['Accuracy'].iloc[0]
                    st.metric("Proposed Majority Voting", f"{voting_acc*100:.2f}%")
                    st.caption("IEEE Target: 99.4%")
                else:
                    st.metric("Proposed Majority Voting", "N/A")
            with col4:
                best_acc = metrics_df['Accuracy'].max()
                st.metric("Best Model", f"{best_acc*100:.2f}%")
            
            st.markdown("---")
            
            # Traditional Models vs XGBoost Individual vs Proposed XGBoost Majority Voting Comparison
            st.markdown('<h3 class="sub-header">📈 Traditional Models vs XGBoost Individual vs Proposed XGBoost Majority Voting</h3>', unsafe_allow_html=True)
            
            # Create comparison chart
            comparison_data = []
            for _, row in metrics_df.iterrows():
                if row['Model'] in traditional_models:
                    model_type = "Traditional Models"
                elif row['Model'] in xgboost_individual:
                    model_type = "XGBoost Individual"
                elif row['Model'] in xgboost_majority_voting:
                    model_type = "Proposed XGBoost Majority Voting"
                else:
                    model_type = "Other"
                comparison_data.append({
                    'Model': row['Model'],
                    'Accuracy': row['Accuracy'],
                    'Type': model_type
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df,
                x='Model',
                y='Accuracy',
                color='Type',
                title='Traditional Models vs XGBoost Individual vs Proposed XGBoost Majority Voting',
                color_discrete_map={
                    'Traditional Models': '#4ECDC4',
                    'XGBoost Individual': '#96CEB4',
                    'Proposed XGBoost Majority Voting': '#FF4757'
                },
                text='Accuracy'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.add_hline(y=0.994, line_dash="dash", line_color="red", 
                         annotation_text="IEEE Target: 99.4%", 
                         annotation_position="right")
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                yaxis_title='Accuracy',
                legend=dict(title="Model Type")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Comprehensive metrics comparison (Accuracy, Recall, F1-Score per IEEE)
            st.markdown('<h3 class="sub-header">Comprehensive Metrics Comparison</h3>', unsafe_allow_html=True)
            
            metrics_to_plot = ['Accuracy', 'Recall', 'F1-Score']
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy', 'Recall', 'F1-Score', ''),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            traditional_models_list = ['SVM', 'KNN', 'Naive_Bayes', 'AdaBoost']
            positions = [(1,1), (1,2), (2,1)]
            for metric, pos in zip(metrics_to_plot, positions):
                if metric not in metrics_df.columns:
                    continue
                colors = []
                for model in metrics_df['Model']:
                    if model in traditional_models_list:
                        colors.append('#4ECDC4')
                    elif model == 'XGBoost_Individual':
                        colors.append('#96CEB4')
                    elif model == 'XGBoost_Majority_Voting':
                        colors.append('#FF4757')
                    else:
                        colors.append('#FFEAA7')
                fig.add_trace(
                    go.Bar(
                        x=metrics_df['Model'],
                        y=metrics_df[metric],
                        name=metric,
                        marker_color=colors,
                        text=metrics_df[metric].round(4),
                        textposition='outside'
                    ),
                    row=pos[0], col=pos[1]
                )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                title_text="Comprehensive Model Performance (Accuracy, Recall, F1-Score per IEEE)"
            )
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics table with categorization
            st.markdown('<h3 class="sub-header">Detailed Performance Metrics</h3>', unsafe_allow_html=True)
            
            metrics_df_display = metrics_df.copy()
            metrics_df_display['Model Type'] = metrics_df_display['Model'].apply(
                lambda x: 'Traditional Models' if x in traditional_models_list else ('XGBoost Individual' if x == 'XGBoost_Individual' else 'Proposed XGBoost Majority Voting')
            )
            display_cols = ['Model Type', 'Model', 'Accuracy', 'Recall', 'F1-Score']
            if 'AUC' in metrics_df_display.columns:
                display_cols.append('AUC')
            metrics_df_display = metrics_df_display[[c for c in display_cols if c in metrics_df_display.columns]]
            st.dataframe(metrics_df_display.round(4), use_container_width=True)
            
            best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
            st.markdown(f"""
            <div class="metric-card">
                <h3>Best Performing Model</h3>
                <p><strong>{best_model['Model']}</strong> with <strong>{best_model['Accuracy']:.4f}</strong> accuracy</p>
            </div>
            """, unsafe_allow_html=True)

# Disease Prediction Page
elif page == "Disease Prediction":
    st.markdown('<h2 class="sub-header">Liver Cirrhosis Disease Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first in the Model Training page.")
    else:
        st.info("Enter patient information below to get a disease prediction.")
        
        # Create input form - All IEEE paper parameters
        with st.form("prediction_form"):
            st.markdown('<h3 class="sub-header">Patient Information - All IEEE Paper Parameters</h3>', unsafe_allow_html=True)
            st.info("**All parameters as per IEEE paper:** Age, Sex, Drug, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin, Triglycerides, Platelets, Cholesterol, Albumin, Copper, Alkaline Phosphatase, SGOT")
            
            # Demographics and Basic Info
            st.markdown('<h4 class="sub-header">Demographics</h4>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age (years)", min_value=0, max_value=120, value=50, help="Patient age")
            with col2:
                sex = st.selectbox("Sex", ["Female", "Male"], help="Gender (0 = Female, 1 = Male)")
            with col3:
                drug = st.selectbox("Drug", ["No", "Yes"], help="Medication usage (0 = No, 1 = Yes)")
            
            # Disease State Parameters
            st.markdown('<h4 class="sub-header">Disease State Parameters</h4>', unsafe_allow_html=True)
            col4, col5, col6, col7 = st.columns(4)
            with col4:
                ascites = st.selectbox("Ascites", ["No", "Yes"], help="Presence of fluid in abdomen (0 = No, 1 = Yes)")
            with col5:
                hepatomegaly = st.selectbox("Hepatomegaly", ["No", "Yes"], help="Enlarged liver (0 = No, 1 = Yes)")
            with col6:
                spiders = st.selectbox("Spiders", ["No", "Yes"], help="Spider angiomas (0 = No, 1 = Yes)")
            with col7:
                edema = st.selectbox("Edema", ["No", "Yes"], help="Swelling (0 = No, 1 = Yes)")
            
            # Biochemical Features - Part 1 (IEEE Paper: Feature Ranges)
            st.markdown('<h4 class="sub-header">Biochemical Features (Feature Ranges per IEEE Paper)</h4>', unsafe_allow_html=True)
            st.caption("**Note:** The model classifies whether values fall within or outside typical ranges as specified in the IEEE paper.")
            col8, col9 = st.columns(2)
            
            with col8:
                # Bilirubin
                bilirubin = st.number_input("Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=1.0, step=0.1,
                                           help="IEEE Paper Range: Dataset includes ranges of bilirubin. Normal: 0.1-1.2 mg/dL")
                # Triglycerides
                triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=0.0, max_value=500.0, value=120.0, step=0.1, 
                                                help="IEEE Paper Range: Dataset includes ranges of triglycerides. Normal: 50-150 mg/dL")
                # Platelets
                platelets = st.number_input("Platelets (10^9/L)", min_value=0, max_value=1000, value=300,
                                           help="IEEE Paper Range: Dataset includes ranges of platelets. Normal: 150-450 x 10^9/L")
                # Cholesterol
                cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0.0, max_value=400.0, value=180.0, step=0.1,
                                             help="IEEE Paper Range: Dataset includes ranges of cholesterol. Normal: <200 mg/dL")
            
            with col9:
                # Albumin
                albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, value=4.0, step=0.1,
                                         help="IEEE Paper Range: Dataset includes ranges of albumin. Normal: 3.5-5.0 g/dL")
                # Copper
                copper = st.number_input("Copper (mcg/dL)", min_value=0.0, max_value=300.0, value=100.0, step=0.1,
                                        help="IEEE Paper Range: Dataset includes ranges of copper. Normal: 70-140 mcg/dL")
                # Alkaline Phosphatase
                alkaline_phosphatase = st.number_input("Alkaline Phosphatase (U/L)", min_value=0, max_value=2000, value=100,
                                                       help="IEEE Paper Range: Dataset includes ranges of alkaline phosphatase. Normal: 44-147 U/L")
                # SGOT (AST)
                sgot = st.number_input("SGOT / AST (U/L)", min_value=0, max_value=1000, value=30,
                                       help="IEEE Paper Range: Dataset includes ranges of SGOT. Normal: 10-40 U/L") 
            
            submitted = st.form_submit_button("🔬 Predict Disease", type="primary")
            
            if submitted:
                try:
                    # Prepare input data - All IEEE paper parameters
                    input_data_model = {
                        'Age': age,
                        'Sex': 1 if sex == 'Male' else 0,
                        'Drug': 1 if drug == 'Yes' else 0,
                        'Ascites': 1 if ascites == 'Yes' else 0,
                        'Hepatomegaly': 1 if hepatomegaly == 'Yes' else 0,
                        'Spiders': 1 if spiders == 'Yes' else 0,
                        'Edema': 1 if edema == 'Yes' else 0,
                        'Bilirubin': bilirubin,
                        'Triglycerides': triglycerides,
                        'Platelets': platelets,
                        'Cholesterol': cholesterol,
                        'Albumin': albumin,
                        'Copper': copper,
                        'Alkaline_Phosphatase': alkaline_phosphatase,
                        'SGOT': sgot  # SGOT is Aspartate Aminotransferase (AST)
                    }
                    
                    # Stage data for classification display
                    stage_data = {
                        'Ascites': input_data_model['Ascites'],
                        'Hepatomegaly': input_data_model['Hepatomegaly'],
                        'Spiders': input_data_model['Spiders'],
                        'Edema': input_data_model['Edema']
                    }
                    
                    # Use Proposed XGBoost Majority Voting Model (Internal Ensemble Technique)
                    if 'results' not in st.session_state or 'XGBoost_Majority_Voting' not in st.session_state.results:
                        st.warning("No trained model found. Please train models first on the Model Training page.")
                        st.stop()
                    if st.session_state.scaler is None or st.session_state.feature_columns is None:
                        st.warning("Preprocessor state not found. Please train models again.")
                        st.stop()
                    
                    try:
                        model = st.session_state.results['XGBoost_Majority_Voting']['model']
                        scaler = st.session_state.scaler
                        feature_columns = st.session_state.feature_columns
                        X_input = pd.DataFrame([input_data_model])[feature_columns]
                        X_scaled = scaler.transform(X_input)
                        prediction = int(model.predict(X_scaled)[0])
                        proba = model.predict_proba(X_scaled)[0]
                        confidence = float(proba[1] if prediction == 1 else proba[0])
                        risk_level = 'High' if prediction == 1 else 'Low'
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                        import traceback
                        with st.expander("Details"):
                            st.code(traceback.format_exc())
                        st.stop()
                    
                    st.markdown('<h3 class="sub-header">Prediction Results (Proposed XGBoost Majority Voting Model)</h3>', unsafe_allow_html=True)
                    st.caption("Internal Ensemble Technique within XGBoost - not a VotingClassifier across different algorithms.")
                    
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="danger-card">
                            <h2>Liver Cirrhosis Detected</h2>
                            <p><strong>Confidence Score:</strong> {confidence:.2%}</p>
                            <p><strong>Risk Level:</strong> High</p>
                            <p>Please consult with a healthcare professional immediately.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>No Cirrhosis Detected</h2>
                            <p><strong>Confidence Score:</strong> {confidence:.2%}</p>
                            <p><strong>Risk Level:</strong> Low</p>
                            <p>Continue regular health monitoring.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Stage Classification Display (Ascites / Hepatomegaly / Spiders / Edema per IEEE)
                    st.markdown('<h3 class="sub-header">Liver Cirrhosis Stage Classification</h3>', unsafe_allow_html=True)
                    st.markdown("**Stage indicators (per IEEE paper): Ascites, Hepatomegaly, Spiders, Edema**")
                    indicators_df = pd.DataFrame({
                        'Indicator': ['Ascites', 'Hepatomegaly', 'Spiders', 'Edema'],
                        'Status': [
                            'Present' if stage_data['Ascites'] == 1 else 'Absent',
                            'Present' if stage_data['Hepatomegaly'] == 1 else 'Absent',
                            'Present' if stage_data['Spiders'] == 1 else 'Absent',
                            'Present' if stage_data['Edema'] == 1 else 'Absent'
                        ],
                        'Value': [
                            stage_data['Ascites'],
                            stage_data['Hepatomegaly'],
                            stage_data['Spiders'],
                            stage_data['Edema']
                        ]
                    })
                    st.dataframe(indicators_df, use_container_width=True, hide_index=True)
                    
                    stage_score = stage_data['Ascites'] + stage_data['Hepatomegaly'] + stage_data['Spiders'] + stage_data['Edema']
                    if prediction == 0:
                        stage_label = 'No Disease'
                    else:
                        if stage_score <= 1:
                            stage_label = 'Early Stage'
                        elif stage_score == 2:
                            stage_label = 'Moderate Stage'
                        else:
                            stage_label = 'Advanced Stage'
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Predicted Stage: {stage_label}</h3>
                        <p><strong>Stage Score:</strong> {stage_score} out of 4 indicators (Ascites, Hepatomegaly, Spiders, Edema)</p>
                        <p><em>Per IEEE ICMLAS-2025 Paper Methodology</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Performance Dashboard Page
elif page == "Performance Dashboard":
    st.markdown('<h2 class="sub-header">Performance Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first in the Model Training page.")
    else:
        st.success("Performance dashboard is ready!")
        
        # Key metrics
        st.markdown('<h3 class="sub-header">Key Performance Indicators</h3>', unsafe_allow_html=True)
        
        if 'evaluation_results' in st.session_state:
            metrics_df = st.session_state.evaluation_results['metrics_df']
            
            # Calculate summary statistics
            best_accuracy = metrics_df['Accuracy'].max()
            avg_accuracy = metrics_df['Accuracy'].mean()
            best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Accuracy", f"{best_accuracy:.4f}")
            with col2:
                st.metric("Average Accuracy", f"{avg_accuracy:.4f}")
            with col3:
                st.metric("Best Model", best_model)
            with col4:
                st.metric("Total Models", len(metrics_df))
            
            # Traditional Models vs XGBoost Individual vs Proposed XGBoost Majority Voting
            traditional_models_list = ['SVM', 'KNN', 'Naive_Bayes', 'AdaBoost']
            
            # Performance comparison chart
            st.markdown('<h3 class="sub-header">Model Performance Comparison</h3>', unsafe_allow_html=True)
            
            comparison_data = []
            for _, row in metrics_df.iterrows():
                if row['Model'] in traditional_models_list:
                    model_type = "Traditional Models"
                elif row['Model'] == 'XGBoost_Individual':
                    model_type = "XGBoost Individual"
                elif row['Model'] == 'XGBoost_Majority_Voting':
                    model_type = "Proposed XGBoost Majority Voting"
                else:
                    model_type = "Other"
                comparison_data.append({
                    'Model': row['Model'],
                    'Accuracy': row['Accuracy'],
                    'Type': model_type
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df, 
                x='Model', 
                y='Accuracy',
                title='Traditional Models vs XGBoost Individual vs Proposed XGBoost Majority Voting',
                color='Type',
                color_discrete_map={
                    'Traditional Models': '#4ECDC4',
                    'XGBoost Individual': '#96CEB4',
                    'Proposed XGBoost Majority Voting': '#FF4757'
                },
                text='Accuracy'
            )
            fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
            fig.add_hline(y=0.994, line_dash="dash", line_color="red", 
                         annotation_text="IEEE Target: 99.4%", 
                         annotation_position="right")
            fig.update_layout(
                xaxis_tickangle=-45,
                legend=dict(title="Model Type")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<h3 class="sub-header">Detailed Performance Metrics</h3>', unsafe_allow_html=True)
            
            metrics_df_display = metrics_df.copy()
            metrics_df_display['Model Type'] = metrics_df_display['Model'].apply(
                lambda x: 'Traditional Models' if x in traditional_models_list else ('XGBoost Individual' if x == 'XGBoost_Individual' else 'Proposed XGBoost Majority Voting')
            )
            display_cols = ['Model Type', 'Model', 'Accuracy', 'Recall', 'F1-Score']
            if 'AUC' in metrics_df_display.columns:
                display_cols.append('AUC')
            metrics_df_display = metrics_df_display[[c for c in display_cols if c in metrics_df_display.columns]]
            st.dataframe(metrics_df_display.round(4), use_container_width=True)
            
            st.info("**IEEE ICMLAS-2025 Reported Accuracy:** 99.4% for Proposed XGBoost Majority Voting Technique")
            
            st.markdown('<h3 class="sub-header">Model Performance Radar Chart</h3>', unsafe_allow_html=True)
            
            radar_metrics = ['Accuracy', 'Recall', 'F1-Score']
            fig = go.Figure()
            for _, row in metrics_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in radar_metrics if m in metrics_df.columns],
                    theta=radar_metrics,
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Radar Chart"
            )
            
            st.plotly_chart(fig, use_container_width=True)

# About Page (second and only other page)
elif page == "About":
    st.markdown('<h2 class="sub-header">About This Project</h2>', unsafe_allow_html=True)
    st.markdown("""
**Project Title:** Prediction of Liver Cirrhosis Disease Using XGBoost Majority Voting Technique in Machine Learning

**Objective:** Build a machine learning system to predict liver cirrhosis (0 = absent, 1 = present) and classify its stages using an XGBoost model enhanced with an **Internal Majority Voting Technique**, and compare it with **Traditional Models** (SVM, KNN, Naïve Bayes, AdaBoost).

**Input Parameters (IEEE):**
- Age, Sex, Drug, Ascites, Hepatomegaly, Spiders, Edema, Bilirubin
- Ranges of: Triglycerides, Platelets, Cholesterol, Albumin, Copper, Alkaline Phosphatase, SGOT, Bilirubin

**Disease Stage Classification:** Ascites, Hepatomegaly, Spiders, Edema

**Evaluation Metrics:** Accuracy, F1-Score, Recall

**Reported Benchmark:** Proposed XGBoost Majority Voting Accuracy ≈ 99.4%
""")

    if st.session_state.models_trained and 'evaluation_results' in st.session_state:
        metrics_df = st.session_state.evaluation_results.get('metrics_df')
        if metrics_df is not None and not metrics_df.empty:
            st.markdown('<h3 class="sub-header">Loaded Model Metrics</h3>', unsafe_allow_html=True)
            st.dataframe(metrics_df[['Model', 'Accuracy', 'Recall', 'F1-Score']].round(4), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <p><strong>Liver Cirrhosis Disease Prediction System</strong></p>
    <p>Using XGBoost Majority Voting Technique in Machine Learning</p>
    <p>Developed for Final Year Project - SV College of Engineering, Tirupati</p>
</div>
""", unsafe_allow_html=True)

