# BankAssist

### Intelligent Banking Query Classification System

BankAssist is a machine learning-based application that classifies banking-related customer queries into predefined intent categories using Natural Language Processing (NLP).

The system uses TF-IDF for text feature extraction and Logistic Regression for multi-class classification. It provides an interactive interface where users can enter banking queries and view predicted intents along with probability estimates.

## 1. Project Overview

The objective of BankAssist is to automatically identify the intent behind customer banking queries, helping organise and categorise customer support requests.

**Example predictions:**

| Customer Query                           | Predicted Intent                   |
| ---------------------------------------- | ---------------------------------- |
| My card was stolen                       | lost_or_stolen_card                |
| I want to close my account               | terminate_account                  |
| My contactless payment isn't working     | contactless_not_working            |
| My transfer hasn't reached the recipient | transfer_not_received_by_recipient |

## 2. Key Features

* Classifies banking queries into 77 intent categories.
* Extracts text features using TF-IDF.
* Uses Logistic Regression for classification.
* Displays prediction probability estimates.
* Shows the top three predicted intents.
* Provides an interactive Streamlit interface.
* Includes confidence-based prediction warnings.

## 3. Technologies Used

| Technology          | Purpose                               |
| ------------------- | ------------------------------------- |
| Python              | Core programming language             |
| Pandas              | Data loading and exploration          |
| Scikit-learn        | Machine learning and model evaluation |
| TF-IDF              | Text feature extraction               |
| Logistic Regression | Intent classification                 |
| Streamlit           | Interactive web application           |
| Joblib              | Model serialization                   |

## 4. Dataset

**BANKING77 Dataset**

The project uses the BANKING77 dataset introduced by PolyAI.

| Dataset Information |                                 Details |
| ------------------- | --------------------------------------: |
| Training samples    |                                  10,003 |
| Testing samples     |                                   3,080 |
| Intent categories   |                                      77 |
| Features            | Customer query text and intent category |

## 5. Machine Learning Workflow

1. Dataset loading and exploration.
2. Data quality checks.
3. Text feature extraction using TF-IDF.
4. Model training using Logistic Regression.
5. Hyperparameter experimentation.
6. Model evaluation using classification metrics.
7. Error analysis to identify misclassified intents.
8. Model serialization using Joblib.
9. Application development using Streamlit.

## 6. Model Performance

The Logistic Regression model was evaluated using the provided test dataset.

| Evaluation Metric | Result |
| ----------------- | -----: |
| Training Accuracy | 98.18% |
| Testing Accuracy  | 90.00% |

Error analysis was performed to identify frequently confused intent categories and understand the limitations of the classification model.

## 7. Installation and Execution

### Prerequisites

* Python 3.10 or later
* Git
* pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/BankAssist.git
cd BankAssist
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Prepare the Dataset

Obtain the BANKING77 dataset from its original source, subject to its licence and usage terms.

Place the CSV files inside the `data/` directory:

```text
data/
├── train.csv
└── test.csv
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser.

## 8. Project Structure

```text
BankAssist/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   └── bankassist_model.pkl
│
├── notebooks/
│   └── 01_dataset_exploration.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 9. Limitations

* TF-IDF primarily relies on word-level patterns and may struggle with semantically similar queries.
* Closely related banking intents can sometimes be misclassified.
* Prediction probabilities are estimates and are not guaranteed measures of reliability.
* The application is a portfolio prototype and is not intended for real banking decisions.

## 10. Future Improvements

* Experiment with transformer-based NLP models.
* Improve classification of semantically similar intents.
* Evaluate probability calibration.
* Implement more robust input validation.
* Explore multilingual banking query classification.

## Author

**Anfidha Zayan**

B.Tech Information Technology

Areas of Interest: Artificial Intelligence, Machine Learning and Natural Language Processing.
