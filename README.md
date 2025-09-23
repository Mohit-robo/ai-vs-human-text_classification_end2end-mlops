# 🧠 AI vs Human Text Classifier

A complete **end-to-end ML project** for classifying whether a given
text was written by a **Human (0)** or **AI (1)**.\
This project integrates **data preprocessing, model training, MLflow
tracking, and deployment with Streamlit + Heroku**.

------------------------------------------------------------------------

## 🚀 Features

-   Data preprocessing & cleaning (`src/data.py`)\
-   Model training using SVM (`src/train.py`)\
-   Experiment tracking with **MLflow**\
-   Evaluation utilities (`src/evaluate.py`)\
-   Deployment-ready **Streamlit UI** (`app/main.py`)\
-   CI/CD pipeline with **GitHub Actions + Heroku**\
-   Flexible model loading (local `.pkl` or MLflow registry)

------------------------------------------------------------------------

## 📂 Project Structure

``` bash
AI-VS-HUMAN-TEXT-CLASSIFIER/
│
├── app/                   # Streamlit application
│   ├── main.py            # Streamlit UI entrypoint
│   └── svm_model.pkl      # Saved model (deployed on Heroku)
│
├── artifacts/             # Generated artifacts (logs, temp models, etc.)
│
├── data/                  # Dataset folder
│   ├── ai_vs_human.csv
│   ├── ai_vs_human_cleaned.csv
│   ├── train.csv
│   └── test.csv
│
├── mlruns/                # MLflow tracking logs
│
├── notebooks/             # Jupyter notebooks
│   └── 01_eda.ipynb       # Exploratory Data Analysis
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── data.py            # Data loading & preprocessing
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Model evaluation
│
├── .gitignore             # Git ignore rules
├── .python-version        # Python version pinning (Heroku)
├── Procfile               # Heroku process definition
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
```

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repo and install dependencies:

``` bash
git clone https://github.com/your-username/AI-VS-HUMAN-TEXT-CLASSIFIER.git
cd AI-VS-HUMAN-TEXT-CLASSIFIER

# Create a virtual env
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🏋️ Training the Model

Run the training pipeline:

``` bash
python src/train.py
```

-   Data is loaded from `data/`\
-   Model is trained & logged into **MLflow**\
-   The trained model is exported as `.pkl` (e.g. `app/svm_model.pkl`)

------------------------------------------------------------------------

## 📊 Evaluating the Model

``` bash
python src/evaluate.py
```

-   Loads the trained model\
-   Evaluates on `test.csv`\
-   Prints classification report, confusion matrix, etc.

------------------------------------------------------------------------

## 🎛️ Running Streamlit App (Locally)

``` bash
cd app
streamlit run app/main.py --server.port 8000
```

Go to `http://localhost:8000` to interact with the app.

------------------------------------------------------------------------

## 🌐 Deployment on Heroku

### 1. Login & Create App

``` bash
heroku login
heroku create ai-vs-human-text-classifier
```

### 2. Push Code

``` bash
git push heroku main
```

### 3. Open the App

``` bash
heroku open
```

Another option is to deploy with github repo. Directly connect the github repo with the Heroku app. Every time the code is pushed to a specific branch of the repo, the app will be updated and deployed.

------------------------------------------------------------------------

## 📦 Model Loading Options

By default, the app loads the model from:\
- **Local `.pkl` file** → `app/svm_model.pkl`\

You can configure this in `app/main.py`.

------------------------------------------------------------------------

## 📸 Usage (Streamlit UI)

1.  Open the app in your browser.\
2.  Enter a sentence or paragraph in the text box.\
3.  Click **Classify Text**.\
4.  The model will return:
    -   **Prediction**: AI-generated or Human-written\

### Example:

Input:

    The rapid advancements in AI are shaping the future of work.

Output:

    Prediction: A 

Input:

    I had the best coffee today at my favorite café near the office!

Output:

    Prediction: Human 

------------------------------------------------------------------------

## 📌 Requirements

-   Python `3.8.x` (pinned in `.python-version`)\
-   Dependencies listed in `requirements.txt`

------------------------------------------------------------------------

## ✅ To-Do (Future Enhancements)

-   [ ] Extend CI/CD pipeline with tests

------------------------------------------------------------------------

## 👨‍💻 Author

**Mohit Gaikwad**\
*Stable professional \| Fun & thoughtful \| Loves building ML end-to-end
systems*
