Here are my comprehensive **notes on Module 5 – Model Deployment** of the ML Zoomcamp course.

----------

# 📘 ML Zoomcamp – Module 5 Notes (Model Deployment)

## 5.1 Introduction / Overview

-   This module focuses on **deploying ML models** so they can be used across different systems and applications without rerunning training code.
    
-   **Objective:** Serve models as web services so other systems can send requests and receive predictions.
    
-   **Deployment workflow:**
    
    1.  **Train and save the model.**
        
    2.  **Create an API** to serve predictions.
        
    3.  **Containerize** the service for portability.
        
    4.  **Deploy** it to the cloud for global accessibility.
        
-   Key tools used:
    
    -   Flask (original module) → FastAPI (updated approach)
        
    -   Pipenv or UV for dependency management
        
    -   Docker for containerization
        
    -   AWS Elastic Beanstalk or Fly.io for cloud deployment
        

----------

## 5.2 Saving and Loading Models

-   **Goal:** Use trained models later without retraining.
    
-   **Library:** `pickle`
    
-   **Save model:**
    
    ```python
    import pickle
    with open('model.bin', 'wb') as f_out:
        pickle.dump((dict_vectorizer, model), f_out)
    
    ```
    
-   **Load model:**
    
    ```python
    import pickle
    with open('model.bin', 'rb') as f_in:
        dict_vectorizer, model = pickle.load(f_in)
    
    ```
    
-   Always use `'wb'` for writing binary and `'rb'` for reading binary.
    
-   ⚠️ **Security Warning:** Never unpickle files from untrusted sources.
    

----------

## 5.3 Web Services – Introduction to Flask

-   A **web service** allows different applications to communicate over HTTP.
    
-   Common HTTP methods:
    
    -   **GET** – retrieve data.
        
    -   **POST** – send data (e.g., customer details for prediction).
        
    -   **PUT** – update data.
        
    -   **DELETE** – remove data.
        
-   **Example simple Flask service:**
    
    ```python
    from flask import Flask
    app = Flask('ping')
    
    @app.route('/ping', methods=['GET'])
    def ping():
        return 'PONG'
    
    if __name__ == '__main__':
        app.run(debug=True, host='0.0.0.0', port=9696)
    
    ```
    
    Test via browser or:
    
    ```bash
    curl http://localhost:9696/ping
    
    ```
    

----------

## 5.4 Serving the Churn Model with Flask

-   Extend Flask service to handle **POST** requests for predictions.
    
-   **Steps:**
    
    1.  Load the saved model using `pickle`.
        
    2.  Define a route (`/predict`) that accepts customer data in JSON.
        
    3.  Transform input using `dict_vectorizer`.
        
    4.  Predict churn probability.
        
    5.  Return results as JSON.
        
    
    ```python
    from flask import Flask, request, jsonify
    import pickle
    
    with open('churn-model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    app = Flask('churn')
    
    def predict_single(customer, dv, model):
        X = dv.transform([customer])
        y_pred = model.predict_proba(X)[:, 1]
        return y_pred[0]
    
    @app.route('/predict', methods=['POST'])
    def predict():
        customer = request.get_json()
        prediction = predict_single(customer, dv, model)
        churn = prediction >= 0.5
        result = {
            'churn_probability': float(prediction),
            'churn': bool(churn)
        }
        return jsonify(result)
    
    ```
    
-   **Run Production Server:**
    
    -   Linux/Mac: `gunicorn --bind 0.0.0.0:9696 churn:app`
        
    -   Windows: `waitress-serve --listen=0.0.0.0:9696 churn:app`
        

----------

## 5.5 Virtual Environments with Pipenv / UV

-   **Purpose:** Prevent library version conflicts between projects.
    
-   **Pipenv:**
    
    ```bash
    pip install pipenv
    pipenv install numpy scikit-learn==0.24.1 flask
    pipenv shell
    
    ```
    
    Creates:
    
    -   `Pipfile` – lists dependencies.
        
    -   `Pipfile.lock` – locks exact versions for reproducibility.
        
-   Run web app:
    
    ```bash
    pipenv run gunicorn --bind localhost:9696 churn:app
    
    ```
    
-   **UV (Updated alternative):**
    
    -   Faster dependency manager written in Rust.
        
    -   Commands:
        
        ```bash
        pip install uv
        uv init
        uv add scikit-learn fastapi uvicorn
        uv run python app.py
        
        ```
        
    -   Stores dependencies in `.venv/` and tracks versions in `pyproject.toml`.
        

----------

## 5.6 Environment Management with Docker

-   **Purpose:** Package app + dependencies for consistent deployment.
    
-   **Sample Dockerfile:**
    
    ```dockerfile
    FROM python:3.8.12-slim
    RUN pip install pipenv
    WORKDIR /app
    COPY ["Pipfile", "Pipfile.lock", "./"]
    RUN pipenv install --deploy --system
    COPY ["*.py", "churn-model.bin", "./"]
    EXPOSE 9696
    ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "churn_serving:app"]
    
    ```
    
-   **Build & Run:**
    
    ```bash
    docker build -t churn-prediction .
    docker run -it -p 9696:9696 churn-prediction:latest
    
    ```
    
-   Flags:
    
    -   `-t`: tag
        
    -   `-p`: port mapping
        
    -   `--rm`: remove container after exit
        

----------

## 5.7 Deployment to the Cloud

### Option 1: AWS Elastic Beanstalk

-   AWS-managed platform for app deployment and scaling.
    
-   **Steps:**
    
    ```bash
    pipenv install awsebcli --dev
    eb init -p docker -r eu-north-1 churn-serving
    eb local run --port 9696
    eb create churn-serving-env
    eb terminate churn-serving-env
    
    ```
    
-   **Benefits:**
    
    -   Handles scaling automatically.
        
    -   Integrates easily with Docker containers.
        
-   **Security:** Restrict public access and protect endpoints.
    

----------

### Option 2: Heroku (Alternative)

-   Simple deployment for small Flask/FastAPI apps.
    
-   **Required files:**
    
    -   `requirements.txt`
        
    -   `Procfile`:
        
        ```
        web: gunicorn churn_serving:app
        
        ```
        
    -   Deploy via GitHub or CLI.
        

----------

### Option 3: Fly.io (Updated Module)

-   Lightweight and developer-friendly cloud platform.
    
-   **Workflow:**
    
    1.  Use **FastAPI** instead of Flask.
        
    2.  Package with **Docker**.
        
    3.  Deploy using:
        
        ```bash
        fly launch
        fly deploy
        
        ```
        
-   Benefits:
    
    -   Simpler and faster than AWS Elastic Beanstalk.
        
    -   Supports scaling and edge deployments.
        
    -   Great for personal or small-scale ML APIs.
        

----------

## 5.8 FastAPI (Modern Replacement for Flask)

-   **Advantages:**
    
    -   Async support (faster performance).
        
    -   Automatic API docs at `/docs`.
        
    -   Built-in input validation via **Pydantic**.
        
-   **Example:**
    
    ```python
    from fastapi import FastAPI
    from pydantic import BaseModel
    
    class Customer(BaseModel):
        gender: str
        seniorcitizen: int
        partner: str
        dependents: str
        tenure: int
        monthlycharges: float
        totalcharges: float
    
    app = FastAPI()
    
    @app.get('/ping')
    def ping():
        return {"response": "PONG"}
    
    @app.post('/predict')
    def predict(customer: Customer):
        # Load model and predict
        return {"churn_probability": 0.67, "churn": True}
    
    ```
    
-   Run with:
    
    ```bash
    uvicorn app:app --reload --port 9696
    
    ```
    
-   Visit: `http://localhost:9696/docs` for interactive Swagger UI.
    

----------

## 5.9 Summary

-   **Deployment Pipeline Overview:**
    
    1.  Train → Save model (Pickle)
        
    2.  Serve → Flask or FastAPI API
        
    3.  Isolate → Pipenv or UV virtual environment
        
    4.  Package → Docker container
        
    5.  Deploy → AWS Elastic Beanstalk / Fly.io / Heroku
        
-   **Core Idea:**  
    Build once, deploy anywhere — ensuring reproducibility, scalability, and maintainability.
    

----------

Would you like me to **visualize this as a deployment pipeline diagram (with arrows showing model → API → Docker → Cloud)** for easy inclusion in a presentation or Medium post?
