
import streamlit as st
import pandas as pd
import plotly.express as px
import tensorflow as tf

from pathlib import Path
from PIL import Image

from src.predict import predict_image
from database.db_manager import (
    save_prediction,
    get_prediction_history
)


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgroVision AI | Plant Health Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F6F8F4;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */

[data-testid="stSidebar"] {
    background-color: #173D2C;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #FFFFFF !important;
}

/* Headings */

.main-title {
    color: #173D2C;
    font-size: 32px;
    font-weight: 800;
}

.sub-title {
    color: #718477;
    font-size: 15px;
}

/* Hero container */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5EBE5;
    border-radius: 14px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #718477;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #173D2C;
    font-weight: 800;
}

/* Buttons */

div.stButton > button {
    background-color: #286B49;
    color: #FFFFFF;
    border-radius: 9px;
    border: none;
    font-weight: 600;
    padding: 10px 20px;
}

div.stButton > button:hover {
    background-color: #173D2C;
    color: #FFFFFF;
}

/* File uploader */

[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    border: 1px dashed #A9C5AF;
    border-radius: 12px;
    padding: 15px;
}

/* Tables */

[data-testid="stDataFrame"] {
    border-radius: 10px;
}

/* Divider */

hr {
    border-color: #E5EBE5;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. CONSTANTS
# =========================================================

MODEL_PATH = Path("models/mobilenet_best.keras")
RESULTS_DIR = Path("results")

TEST_ACCURACY = 94.81
TEST_IMAGE_COUNT = 135


# =========================================================
# 4. LOAD MODEL
# =========================================================

@st.cache_resource
def load_prediction_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


# =========================================================
# 5. DATABASE HELPER
# =========================================================

def get_history_dataframe():

    records = get_prediction_history()

    columns = [
        "ID",
        "Image Name",
        "Predicted Class",
        "Confidence",
        "Date & Time"
    ]

    df = pd.DataFrame(records, columns=columns)

    if not df.empty:

        df["Predicted Class"] = (
            df["Predicted Class"]
            .astype(str)
            .str.replace("_", " ", regex=False)
        )

        df["Confidence"] = pd.to_numeric(
            df["Confidence"],
            errors="coerce"
        )

        df["Date & Time"] = pd.to_datetime(
            df["Date & Time"],
            errors="coerce"
        )

    return df


# =========================================================
# 6. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🌿 AgroVision AI")

    st.caption("PLANT HEALTH INTELLIGENCE")

    st.divider()

    st.markdown("### MAIN MENU")

    selected_page = st.radio(
        "Navigate",
        [
            "Home",
            "Disease Detection",
            "Prediction History",
            "Analytics",
            "Model Performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.caption("POWERED BY DEEP LEARNING")

    st.caption(
        "Intelligent Plant Disease Classification System"
    )


# 
# =========================================================
# 7. HOME PAGE
# =========================================================

if selected_page == "Home":

    # Header
    st.title("AgroVision AI")

    st.markdown(
        "### Intelligent Plant Disease Detection and Classification"
    )

    st.write(
        "An AI-powered plant health platform that uses deep learning "
        "to analyze plant leaf images, identify disease categories, "
        "and provide data-driven insights."
    )

    st.divider()

    # Hero Section
    with st.container(border=True):

        st.markdown("## 🌱 Smarter Farming Starts with Healthy Plants")

        st.write(
            "Detect plant diseases through intelligent image classification. "
            "Upload a leaf image and let our trained deep learning model "
            "analyze its condition."
        )

        st.markdown(
            "**AI-Powered Detection**  |  "
            "**Deep Learning**  |  "
            "**Real-Time Prediction**  |  "
            "**Data Analytics**"
        )

        st.info(
            "Navigate to Disease Detection from the sidebar "
            "to analyze a plant leaf image."
        )

    # System Overview
    st.markdown("## System Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Disease Categories",
            value="09"
        )

    with col2:
        st.metric(
            label="Best Performing Model",
            value="MobileNetV2"
        )

    with col3:
        st.metric(
            label="Test Accuracy",
            value=f"{TEST_ACCURACY:.2f}%"
        )

    st.divider()

    # Platform Capabilities
    st.markdown("## Platform Capabilities")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):
            st.subheader("🌿 Disease Detection")

            st.write(
                "Analyze uploaded plant leaf images and classify "
                "them into supported disease categories using "
                "the trained MobileNetV2 model."
            )

        with st.container(border=True):
            st.subheader("📋 Prediction History")

            st.write(
                "Access previously recorded predictions, "
                "classification results, confidence scores, "
                "and timestamps."
            )

    with col2:

        with st.container(border=True):
            st.subheader("📊 Analytics Dashboard")

            st.write(
                "Visualize disease distribution, prediction "
                "statistics, confidence levels, and historical "
                "prediction activity."
            )

        with st.container(border=True):
            st.subheader("🧠 Model Evaluation")

            st.write(
                "Evaluate and compare CNN and MobileNetV2 "
                "using accuracy and other classification metrics."
            )

    st.divider()

    # Project Workflow
    st.markdown("## How AgroVision AI Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 01. Upload")
        st.write(
            "Upload a clear image of a plant leaf "
            "in JPG, JPEG, or PNG format."
        )

    with step2:
        st.markdown("### 02. Analyze")
        st.write(
            "The trained deep learning model processes "
            "the image and predicts its disease category."
        )

    with step3:
        st.markdown("### 03. Explore")
        st.write(
            "View the prediction confidence and explore "
            "results through the analytics dashboard."
        )

    st.divider()

    # Evaluation Information
    st.markdown("## Model Evaluation")

    st.write(
        f"The selected MobileNetV2 model achieved "
        f"**{TEST_ACCURACY:.2f}% accuracy** on "
        f"**{TEST_IMAGE_COUNT} held-out test images**."
    )

    st.caption(
        "Note: This is an experimental evaluation on a limited test dataset. "
        "Performance on real-world field images may vary."
    )

    st.divider()

    st.caption(
        "AgroVision AI | Deep Learning-Based Plant Disease "
        "Detection and Classification"
    )


# =========================================================
# 8. DISEASE DETECTION
# =========================================================

elif selected_page == "Disease Detection":

    st.title("Plant Disease Detection")

    st.write(
        "Upload a plant leaf image to analyze its condition "
        "using our trained MobileNetV2 deep learning model."
    )

    st.divider()

    # Image Upload Section
    st.markdown("## Upload Plant Image")

    with st.container(border=True):

        uploaded_file = st.file_uploader(
            "Choose a plant leaf image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG and PNG."
        )

        st.caption(
            "For better prediction results, upload a clear image "
            "with the plant leaf visible."
        )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file).convert("RGB")

            st.divider()
            st.markdown("## Image Preview")

            col1, col2 = st.columns([1.2, 1])

            with col1:

                st.image(
                    image,
                    caption="Uploaded Plant Leaf",
                    use_container_width=True
                )

            with col2:

                with st.container(border=True):

                    st.markdown("### Image Information")

                    st.write("**File Name**")
                    st.write(uploaded_file.name)

                    st.write("**Image Dimensions**")
                    st.write(f"{image.width} × {image.height} pixels")

                    st.write("**Image Format**")
                    st.write("RGB")

                    st.write("**Model**")
                    st.write("MobileNetV2")

            st.divider()

            # Prediction Button
            predict_button = st.button(
                "Analyze Plant Image",
                type="primary",
                use_container_width=True
            )

            if predict_button:

                with st.spinner(
                    "AI model is analyzing the plant image..."
                ):

                    model = load_prediction_model()

                    result = predict_image(
                        image,
                        model
                    )

                    save_prediction(
                        image_name=uploaded_file.name,
                        predicted_class=result["predicted_class"],
                        confidence=result["confidence"]
                    )

                st.success(
                    "Plant image analysis completed successfully!"
                )

                st.divider()

                # Prediction Results
                st.markdown("## AI Prediction Results")

                with st.container(border=True):

                    st.markdown("### Detected Plant Category")

                    st.subheader(
                        result["predicted_class"].replace("_", " ")
                    )

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:

                        st.metric(
                            label="Prediction Confidence",
                            value=f'{result["confidence"]:.2f}%'
                        )

                    with result_col2:

                        st.metric(
                            label="Classification Model",
                            value="MobileNetV2"
                        )

                    confidence_value = min(
                        max(result["confidence"] / 100, 0),
                        1
                    )

                    st.markdown("**Confidence Level**")

                    st.progress(confidence_value)

                    st.caption(
                        f"Model confidence: {result['confidence']:.2f}%"
                    )

                st.info(
                    "The displayed confidence represents the model's "
                    "predicted probability for the selected class. "
                    "It does not guarantee diagnostic correctness."
                )

                st.caption(
                    "This prediction has been saved to the "
                    "prediction history database."
                )

        except Exception as error:

            st.error(
                f"Unable to process the image: {error}"
            )

    else:

        st.info(
            "Upload a plant leaf image to begin disease detection."
        )


# =========================================================
# 9. PREDICTION HISTORY
# =========================================================

elif selected_page == "Prediction History":

    st.title("Prediction History")

    st.write(
        "Monitor and review previously recorded plant disease "
        "classification results stored in PostgreSQL."
    )

    st.divider()

    try:

        df = get_history_dataframe()

        if df.empty:

            st.info(
                "No prediction records are available yet. "
                "Analyze a plant leaf image to create your first record."
            )

        else:

            # Summary Metrics
            st.markdown("## History Overview")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Predictions",
                    f"{len(df):,}"
                )

            with col2:
                st.metric(
                    "Average Confidence",
                    f'{df["Confidence"].mean():.2f}%'
                )

            with col3:
                unique_classes = df["Predicted Class"].nunique()

                st.metric(
                    "Disease Categories Detected",
                    unique_classes
                )

            st.divider()

            # Search and Filter
            st.markdown("## Prediction Records")

            search = st.text_input(
                "Search Records",
                placeholder="Enter image name or disease category..."
            )

            filtered_df = df.copy()

            if search.strip():

                mask = (
                    filtered_df["Image Name"]
                    .astype(str)
                    .str.contains(
                        search.strip(),
                        case=False,
                        na=False
                    )
                    |
                    filtered_df["Predicted Class"]
                    .astype(str)
                    .str.contains(
                        search.strip(),
                        case=False,
                        na=False
                    )
                )

                filtered_df = filtered_df[mask]

            # Results Table
            if filtered_df.empty:

                st.warning(
                    "No records match your search."
                )

            else:

                display_df = filtered_df.copy()

                display_df["Predicted Class"] = (
                    display_df["Predicted Class"]
                    .astype(str)
                    .str.replace("_", " ", regex=False)
                )

                display_df["Confidence"] = (
                    display_df["Confidence"]
                    .apply(lambda value: f"{float(value):.2f}%")
                )

                display_df["Date & Time"] = (
                    display_df["Date & Time"]
                    .dt.strftime("%d %b %Y, %I:%M %p")
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Image Name": st.column_config.TextColumn(
                            "Image Name"
                        ),
                        "Predicted Class": st.column_config.TextColumn(
                            "Detected Category"
                        ),
                        "Confidence": st.column_config.TextColumn(
                            "Confidence"
                        ),
                        "Date & Time": st.column_config.TextColumn(
                            "Prediction Date & Time"
                        )
                    }
                )

            st.caption(
                f"Displaying {len(filtered_df)} of {len(df)} "
                "stored prediction records."
            )

    except Exception as error:

        st.error(
            f"Unable to retrieve prediction history: {error}"
        )


# =========================================================
# 10. ANALYTICS DASHBOARD
# =========================================================

elif selected_page == "Analytics":

    st.title("Analytics Dashboard")

    st.write(
        "Explore prediction statistics, disease distribution, "
        "model confidence and historical activity."
    )

    st.divider()

    try:

        df = get_history_dataframe()

        if df.empty:

            st.info(
                "Analytics will appear here after plant disease "
                "predictions have been recorded."
            )

        else:

            # Prepare analytics data
            analytics_df = df.copy()

            analytics_df["Disease"] = (
                analytics_df["Predicted Class"]
                .astype(str)
                .str.replace("_", " ", regex=False)
            )

            analytics_df["Confidence"] = pd.to_numeric(
                analytics_df["Confidence"],
                errors="coerce"
            )

            # Dashboard Metrics
            st.markdown("## Prediction Overview")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Predictions",
                    f"{len(analytics_df):,}"
                )

            with col2:
                average_confidence = analytics_df[
                    "Confidence"
                ].mean()

                st.metric(
                    "Average Confidence",
                    f"{average_confidence:.2f}%"
                    if pd.notna(average_confidence)
                    else "N/A"
                )

            with col3:
                st.metric(
                    "Disease Categories",
                    analytics_df["Disease"].nunique()
                )

            st.divider()

            # Disease Distribution
            st.markdown("## Disease-wise Distribution")

            disease_counts = (
                analytics_df["Disease"]
                .value_counts()
                .rename_axis("Disease")
                .reset_index(name="Predictions")
            )

            fig1 = px.bar(
                disease_counts,
                x="Disease",
                y="Predictions",
                color="Disease",
                text="Predictions",
                template="plotly_white",
                title="Distribution of Predicted Disease Categories"
            )

            fig1.update_traces(
                textposition="outside"
            )

            fig1.update_layout(
                showlegend=False,
                xaxis_title="Disease Category",
                yaxis_title="Number of Predictions",
                title_x=0.02,
                margin=dict(t=65, b=80),
                xaxis_tickangle=-25
            )

            st.plotly_chart(
                fig1,
                use_container_width=True
            )

            st.divider()

            # Confidence Distribution
            st.markdown("## Prediction Confidence Analysis")

            confidence_df = analytics_df.dropna(
                subset=["Confidence"]
            )

            if not confidence_df.empty:

                fig2 = px.histogram(
                    confidence_df,
                    x="Confidence",
                    nbins=10,
                    template="plotly_white",
                    title="Distribution of Model Confidence Scores"
                )

                fig2.update_layout(
                    xaxis_title="Prediction Confidence (%)",
                    yaxis_title="Number of Predictions",
                    title_x=0.02,
                    bargap=0.12
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

            else:

                st.info(
                    "No valid confidence values are available."
                )

            st.divider()

            # Prediction Activity
            st.markdown("## Prediction Activity Over Time")

            activity_df = analytics_df.dropna(
                subset=["Date & Time"]
            ).copy()

            if not activity_df.empty:

                activity_df["Date"] = (
                    activity_df["Date & Time"].dt.date
                )

                daily_counts = (
                    activity_df.groupby("Date")
                    .size()
                    .reset_index(name="Predictions")
                )

                fig3 = px.line(
                    daily_counts,
                    x="Date",
                    y="Predictions",
                    markers=True,
                    template="plotly_white",
                    title="Daily Plant Disease Prediction Activity"
                )

                fig3.update_traces(
                    mode="lines+markers",
                    marker=dict(size=9)
                )

                fig3.update_layout(
                    xaxis_title="Prediction Date",
                    yaxis_title="Number of Predictions",
                    title_x=0.02
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

            else:

                st.info(
                    "No valid timestamps are available "
                    "for activity analysis."
                )

            st.divider()

            st.caption(
                "Analytics are generated dynamically from "
                "prediction records stored in PostgreSQL."
            )

    except Exception as error:

        st.error(
            f"Unable to generate analytics: {error}"
        )



# =========================================================
# 11. MODEL PERFORMANCE
# =========================================================

elif selected_page == "Model Performance":

    st.title("Model Performance Evaluation")

    st.write(
        "Comparative evaluation of CNN and MobileNetV2 "
        "deep learning models for plant disease classification."
    )

    st.divider()

    csv_path = RESULTS_DIR / "model_comparison.csv"
    accuracy_path = RESULTS_DIR / "accuracy_comparison.png"
    metrics_path = RESULTS_DIR / "metrics_comparison.png"

    if csv_path.exists():

        comparison_df = pd.read_csv(csv_path)

        st.markdown("## Model Comparison")

        st.write(
            "The following table presents the evaluation "
            "results of the trained classification models."
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # Display comparison metrics when columns are available
        accuracy_column = next(
            (
                col for col in comparison_df.columns
                if "accuracy" in col.lower()
            ),
            None
        )

        if accuracy_column is not None and len(comparison_df) >= 2:

            accuracy_values = pd.to_numeric(
                comparison_df[accuracy_column]
                .astype(str)
                .str.replace("%", "", regex=False),
                errors="coerce"
            )

            valid_values = accuracy_values.dropna()

            if len(valid_values) >= 2:

                st.divider()

                st.markdown("## Key Evaluation Results")

                metric1, metric2, metric3 = st.columns(3)

                with metric1:
                    st.metric(
                        "CNN Accuracy",
                        f"{valid_values.iloc[0]:.2f}%"
                    )

                with metric2:
                    st.metric(
                        "MobileNetV2 Accuracy",
                        f"{valid_values.iloc[1]:.2f}%"
                    )

                with metric3:
                    improvement = (
                        valid_values.iloc[1]
                        - valid_values.iloc[0]
                    )

                    st.metric(
                        "Accuracy Improvement",
                        f"{improvement:+.2f}",
                        "Percentage points"
                    )

    else:

        st.warning(
            "The model comparison CSV file was not found."
        )

    st.divider()

    # Accuracy Comparison
    st.markdown("## Accuracy Comparison")

    if accuracy_path.exists():

        st.image(
            str(accuracy_path),
            use_container_width=True
        )

    else:

        st.warning(
            "The accuracy comparison chart was not found."
        )

    st.divider()

    # Classification Metrics
    st.markdown("## Classification Metrics")

    st.write(
        "The classification evaluation considers "
        "Precision, Recall and F1-score."
    )

    if metrics_path.exists():

        st.image(
            str(metrics_path),
            use_container_width=True
        )

    else:

        st.warning(
            "The performance metrics chart was not found."
        )

    st.divider()

    # Evaluation Details
    st.markdown("## Evaluation Details")

    info1, info2 = st.columns(2)

    with info1:

        with st.container(border=True):

            st.subheader("Evaluation Dataset")

            st.metric(
                "Held-out Test Images",
                TEST_IMAGE_COUNT
            )

            st.write(
                "The models were evaluated using the same "
                "held-out test dataset."
            )

    with info2:

        with st.container(border=True):

            st.subheader("Selected Model")

            st.markdown("### MobileNetV2")

            st.write(
                "MobileNetV2 achieved the higher test accuracy "
                "in the current model comparison."
            )

    st.info(
        "The reported results are based on a limited held-out "
        "test dataset. Performance on independent real-world "
        "field images may vary."
    )

    st.caption(
        "AgroVision AI | Deep Learning Model Evaluation"
    )
