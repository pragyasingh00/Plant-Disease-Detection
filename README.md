
# AgroVision AI
### Deep Learning-Based Plant Disease Detection and Classification

AgroVision AI is an intelligent plant health platform that uses deep learning and computer vision to identify plant disease categories from leaf images.

The application uses a trained MobileNetV2 model to classify uploaded plant images and provides prediction confidence, prediction history, and interactive analytics.

## Key Features

- **AI-Based Disease Detection:** Classifies plant leaf images using MobileNetV2.
- **Image Upload:** Supports JPG, JPEG, and PNG formats.
- **Prediction Confidence:** Displays the model's confidence score.
- **Prediction History:** Stores prediction records in PostgreSQL.
- **Analytics Dashboard:** Visualizes disease distribution, confidence scores, and prediction activity.
- **Model Comparison:** Compares CNN and MobileNetV2 performance.
- **Interactive Interface:** Built using Streamlit.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Model Architecture | MobileNetV2, CNN |
| Image Processing | Pillow |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly |
| Frontend | Streamlit |
| Database | PostgreSQL |
| Development Environment | VS Code |

## Supported Disease Categories

The project currently supports nine classification categories:

1. Pepper Bacterial Spot
2. Pepper Healthy
3. Potato Early Blight
4. Potato Healthy
5. Potato Late Blight
6. Tomato Bacterial Spot
7. Tomato Early Blight
8. Tomato Healthy
9. Tomato Late Blight

## Model Performance

The MobileNetV2 model achieved **94.81% test accuracy** on a held-out test dataset of 135 images.

The project compares two deep learning architectures:

- Convolutional Neural Network (CNN)
- MobileNetV2

Evaluation includes accuracy, precision, recall, and F1-score.

*Note: This is an experimental result on a limited test dataset. Performance on independent real-world field images may vary.*

## Application Workflow

1. Upload a plant leaf image.
2. The application preprocesses the image.
3. MobileNetV2 generates a disease classification.
4. The application displays the predicted category and confidence.
5. The prediction is stored in PostgreSQL.
6. Users can explore prediction history and analytics.

## Project Structure

```text
Plant-Disease-Detection/
│
├── app.py
├── src/
│   └── predict.py
│
├── database/
│   ├── db_manager.py
│   └── test_connection.py
│
├── models/
│   ├── mobilenet_best.keras
│   ├── mobilenet_final.keras
│   ├── cnn_best.keras
│   └── cnn_final.keras
│
├── dataset/
│
├── results/
│   ├── model_comparison.csv
│   ├── accuracy_comparison.png
│   └── metrics_comparison.png
│
├── .streamlit/
│   └── secrets.toml
│
├── requirements.txt
└── README.md
```

*The structure above is representative. Update it if your actual project folders differ.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Plant-Disease-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure PostgreSQL

Create a PostgreSQL database and configure the required connection credentials locally.

Do not upload database passwords or private credentials to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Future Scope

- Expand the dataset with more plant species and disease categories.
- Improve model generalization using independent field images.
- Add disease-specific prevention and treatment information.
- Develop a mobile-friendly application.
- Integrate weather and environmental data.
- Explore real-time plant monitoring capabilities.

## Project Status

The current application includes image-based disease classification, PostgreSQL prediction storage, analytics, and model performance comparison.

## Disclaimer

AgroVision AI is an experimental academic project. Its predictions are intended for educational and research purposes and should not replace expert agricultural diagnosis.

## Author

Developed as an academic project in Computer Science and Engineering (Data Science).
