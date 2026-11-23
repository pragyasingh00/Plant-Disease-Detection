# AgroVision AI
### Deep Learning-Based Plant Disease Detection and Classification

AgroVision AI is an AI-powered plant disease detection and classification system developed using Machine Learning, Deep Learning, and Computer Vision.

The application uses a trained **MobileNetV2 model** to identify plant disease categories from uploaded leaf images. It also provides prediction confidence, prediction history, interactive analytics, and model performance comparison.

The project is developed as an academic project in Computer Science and Engineering (Data Science).

---

## Key Features

- **AI-Based Disease Detection:** Identifies plant disease categories using a trained MobileNetV2 model.
- **Image Upload:** Supports JPG, JPEG, and PNG images.
- **Prediction Confidence:** Displays the model's prediction confidence score.
- **Prediction History:** Stores previous prediction records in PostgreSQL.
- **Analytics Dashboard:** Visualizes prediction activity, disease distribution, and confidence scores.
- **Model Performance:** Provides comparison between CNN and MobileNetV2 models.
- **Interactive Interface:** User-friendly application developed using Streamlit.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow / Keras |
| Model Architectures | CNN, MobileNetV2 |
| Image Processing | Pillow |
| Data Processing | NumPy, Pandas |
| Data Visualization | Plotly |
| Frontend | Streamlit |
| Database | PostgreSQL |
| Development Environment | VS Code |

---

## Supported Disease Categories

The current model supports nine classification categories:

| Plant | Disease Category |
|---|---|
| Pepper | Bacterial Spot |
| Pepper | Healthy |
| Potato | Early Blight |
| Potato | Healthy |
| Potato | Late Blight |
| Tomato | Bacterial Spot |
| Tomato | Early Blight |
| Tomato | Healthy |
| Tomato | Late Blight |

---

## Model Performance

The trained MobileNetV2 model achieved **94.81% test accuracy** on a held-out test dataset containing 135 images.

The project evaluates two deep learning architectures:

1. Convolutional Neural Network (CNN)
2. MobileNetV2

Evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score

**Note:** These results are based on a limited experimental test dataset. Performance on independent real-world agricultural images may vary.

---

## Application Workflow

1. User uploads a plant leaf image.
2. The image is converted to RGB and resized to 224 × 224 pixels.
3. MobileNetV2 preprocessing is applied.
4. The trained model predicts the disease category.
5. The application displays the predicted category and confidence.
6. The prediction record is stored in PostgreSQL.
7. Users can view prediction history and explore analytics.

---

## Project Structure

```text
Plant-Disease-Detection/
│
├── .streamlit/
│   └── config.toml
│
├── database/
│   ├── db_manager.py
│   └── test_connection.py
│
├── dataset/
│   └── processed_900/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/
│   ├── mobilenet_best.keras
│   └── mobilenet_final.keras
│
├── results/
│   ├── model_comparison.csv
│   └── evaluation charts and reports
│
├── src/
│   └── predict.py
│
├── app.py
├── download_dataset.py
├── prepare_dataset.py
├── requirements.txt
├── test_data_loader.py
├── .gitignore
└── README.md
```

Additional files may be present in the project directories.

**Important:** The application uses `models/mobilenet_best.keras` for prediction. The dataset's `train` directory is also required because the application obtains its class names from the training subdirectories.

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/pragyasingh00/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure PostgreSQL

Install PostgreSQL and create a database named:

```text
plant_disease_db
```

Create the required table using the SQL provided in the **Database Setup** section below.

Then configure your local Streamlit secrets file:

```text
.streamlit/secrets.toml
```

Use your own PostgreSQL credentials. Never upload this file containing real credentials to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## Database

PostgreSQL is used to store and retrieve prediction history.

The database stores:
- Prediction ID
- Uploaded image filename
- Predicted disease category
- Confidence score
- Prediction date and time

Each team member must configure their own PostgreSQL database and local credentials.

Refer to the Database Setup instructions in this README for the complete configuration.

---

## Future Scope

- Expand the dataset with additional plant species and disease categories.
- Improve model generalization using independent field images.
- Add disease-specific prevention and treatment recommendations.
- Develop a mobile-friendly application.
- Integrate weather and environmental data.
- Explore real-time plant monitoring capabilities.

---

## Project Status

The current application includes:

- Image-based plant disease classification
- MobileNetV2 model integration
- PostgreSQL prediction history
- Interactive analytics dashboard
- CNN and MobileNetV2 performance comparison

---

## Disclaimer

AgroVision AI is an experimental academic project developed for educational and research purposes. Its predictions should not replace professional agricultural diagnosis or expert recommendations.

---

## Repository

GitHub: https://github.com/pragyasingh00/Plant-Disease-Detection

Developed as an academic project in Computer Science and Engineering (Data Science).
