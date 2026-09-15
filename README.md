# Heart Disease Prediction

## Project Overview
''' This project predicts whether a person is likely to have heart disease using Machine Learning.

### Technologies Used

-**Python**
-**NumPy**
-**Pandas**
-**Scikit-learn**
-**MLflow**
-**FastAPI**
-**Streamlit**
-**Docker**

### Project Structure

Heart_Disease_Project/
├── Backend/
│   ├── main.py
│   └── schemas.py
├── Frontend/
│   └── app.py
├── model/
│   └── model.skops
├── data/
├── src/
│   ├── train.py
│   ├── preprocess.py
│   └── utils.py
├── notebook/
│   └── heart_analysis.ipynb
├── docker-compose.yml
└── README.md

### How to Run

**MLflow**

```bash
python -m mlflow ui
```

**Backend**

```bash
python -m uvicorn main:app --reload
```

**Frontend**

```bash
streamlit run app.py
```

**Docker**

```bash
docker compose up --build
```

## Project Features

- Heart disease prediction using Machine Learning
- FastAPI backend
- Streamlit frontend
- Docker containerization
- MLflow experiment tracking