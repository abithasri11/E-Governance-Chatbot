# ML-Enabled E-Governance Citizen Chatbot

An ML-enabled chatbot designed to assist citizens in understanding and accessing Tamil Nadu e-Governance / e-Sevai services.

The system uses **Machine Learning classification** to identify the relevant service from a citizen's query and **Retrieval-Augmented Generation (RAG)** techniques to retrieve relevant government service information.

## 🚀 Live Dashboard

**Streamlit Dashboard:**
https://e-governance-chatbot-p6zndvqzuha5qxlzbcrzf6.streamlit.app/

---

## 📌 Project Overview

Citizens may have difficulty finding information about government services such as service charges, required documents, application procedures and application status.

This project provides a simple chatbot interface where a citizen can enter a question related to Tamil Nadu e-Sevai services.

The system:

1. Accepts a citizen's question.
2. Predicts the relevant e-Sevai service using Machine Learning.
3. Retrieves relevant information from the collected government service data.
4. Displays the retrieved information through a Streamlit dashboard.
5. Provides technical information such as the predicted service and classification confidence.

---

## ✨ Features

* 🤖 ML-based service classification
* 🔎 RAG-based information retrieval
* 📚 Government-service knowledge base
* 🧠 TF-IDF text representation
* 📊 Logistic Regression classification
* 🔬 Hyperparameter tuning
* 📈 Accuracy, Precision, Recall and F1 Score evaluation
* 🖥️ Interactive Streamlit dashboard
* 🏛️ Information collected from official Tamil Nadu e-Sevai sources
* 💬 Citizen-friendly responses

---

## 🏗️ System Architecture

```text
                 Citizen Question
                       │
                       ▼
              ┌─────────────────┐
              │  ML Classifier  │
              │ TF-IDF +        │
              │ Logistic        │
              │ Regression      │
              └────────┬────────┘
                       │
                 Predicted Service
                       │
                       ▼
              ┌─────────────────┐
              │   RAG System    │
              │ Sentence        │
              │ Transformers +  │
              │ FAISS           │
              └────────┬────────┘
                       │
               Retrieved Information
                       │
                       ▼
              ┌─────────────────┐
              │    Chatbot      │
              └────────┬────────┘
                       │
                       ▼
              Streamlit Dashboard
```

---

## 🛠️ Technologies Used

| Component            | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| User Interface       | Streamlit             |
| Text Representation  | TF-IDF                |
| Machine Learning     | Logistic Regression   |
| Semantic Embeddings  | Sentence Transformers |
| Vector Search        | FAISS                 |
| Data Processing      | Pandas, NumPy         |
| ML Evaluation        | Scikit-learn          |
| Data Format          | CSV, JSON             |

---

## 📂 Project Structure

```text
E-Governance/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── ml_classifier.py
│   ├── rag_system.py
│   └── chatbot.py
│
└── data/
    ├── e_sevai_classification_dataset.csv
    ├── rag_documents.json
    └── faiss_index.bin
```

---

## 📊 Machine Learning

The project uses **TF-IDF** to convert citizen queries into numerical feature vectors.

A **Logistic Regression** classifier is then used to predict the relevant e-Sevai service.

### Hyperparameter Tuning

The following values are evaluated:

**C:**

```text
0.01
0.1
1
10
100
```

**Class Weight:**

```text
None
Balanced
```

A total of **10 hyperparameter configurations** are evaluated.

Each configuration is evaluated using:

* Accuracy
* Weighted Precision
* Weighted Recall
* Weighted F1 Score

The configuration with the highest weighted F1 Score is selected as the final model configuration.

---

## 🔎 RAG System

The Retrieval-Augmented Generation component retrieves relevant government service information.

The system uses:

* **Sentence Transformers** for generating semantic embeddings
* **FAISS** for similarity-based retrieval
* **JSON documents** containing government service information

The retrieval process helps the chatbot provide responses based on the collected government information rather than relying only on the ML classifier.

---

## 🖥️ Streamlit Dashboard

The Streamlit application contains two main sections:

### 🤖 Chatbot

Users can enter questions related to Tamil Nadu e-Sevai services.

The dashboard displays:

* Retrieved service information
* Predicted service
* Classification confidence

### ⚙️ Hyperparameter Tuning

The dashboard displays:

* Hyperparameter configurations
* Accuracy
* Precision
* Recall
* F1 Score
* Best hyperparameters
* Final model evaluation metrics

---

# ▶️ How to Run the Project

## 1. Clone the Repository

Open a terminal or command prompt and run:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Then move into the project folder:

```bash
cd E-Governance
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Required Libraries

Run:

```bash
pip install -r requirements.txt
```

---

## 4. Check the Data Folder

Make sure the required files are present inside the `data` folder:

```text
data/
├── e_sevai_classification_dataset.csv
├── rag_documents.json
└── faiss_index.bin
```

The CSV file is used by the ML classifier.

The JSON file and FAISS index are used by the RAG system.

---

## 5. Run the Streamlit Dashboard

From the project root folder, run:

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local address similar to:

```text
Local URL: http://localhost:8501
```

Open this address in your web browser.

---

## 🌐 Running the Dashboard Online

The application can be deployed using **Streamlit Community Cloud**.

After deployment, Streamlit provides a public URL similar to:

```text
(https://e-governance-chatbot-p6zndvqzuha5qxlzbcrzf6.streamlit.app/)
```

## 📚 Official Data Sources

The project uses official Tamil Nadu e-Sevai sources for collecting government service information.

* Tamil Nadu e-Sevai Citizen Portal
  https://www.tnesevai.tn.gov.in/citizen/

* e-Sevai Service List
  https://tnesevai.tn.gov.in/Pages/EsevaiServiceList.aspx

* e-Sevai Video Tutorials
  https://www.tnesevai.tn.gov.in/citizen/Pages/VideoTutorial.aspx

* e-Sevai FAQ
  https://www.tnesevai.tn.gov.in/Pages/FAQ.aspx

---

## 👥 Team Members

* Abithasri N K
* Hasnaou A

Both team members contributed to the development of the project.

---
