### **Full Notes Summary: Machine Learning Zoomcamp - Session #1**

This session provides a foundational overview of Machine Learning (ML), covering core concepts, methodologies, and the necessary tools for getting started. The lecture is structured to build a conceptual understanding before diving into practical implementation.

---

#### **1.1 Introduction to Machine Learning (ML)**

The core idea of ML is to **learn patterns from data** to make predictions on new, unseen data.
- **Input**: A set of features (e.g., car's year, make, mileage).
- **Output**: A target variable to predict (e.g., car's price).
- The ML algorithm processes the input data and its corresponding targets to learn a function `g` that can map new inputs to accurate outputs.
- This learned function is called the **model**.
- In practice, you provide the model with new feature data (without the target), and it predicts the target value based on the patterns it learned during training.

---

#### **1.2 Rules vs. Machine Learning**

This section contrasts traditional **rule-based systems** with **Machine Learning**.
- **Rule-Based Systems**: Require humans to manually define logical rules (if-then statements) to classify or predict outcomes. For example:
  - If sender = "promotions@online.com", then classify as "spam".
  - If title contains "tax review" AND sender domain is "online.com", then "spam".
  - These rules become complex and hard to maintain as the problem scales.
- **Machine Learning**: Instead of writing rules, you provide the algorithm with labeled examples (data + target). The algorithm automatically discovers the underlying patterns and relationships to create the model.
- The process involves converting raw data (like email text) into numerical features (e.g., using 0s and 1s to represent the presence or absence of certain words or domains) and then training the model on these features and their corresponding labels (target).

---

#### **1.3 Supervised Machine Learning**

Supervised learning is the most common type of ML, where the model learns from **labeled data**.
- The goal is to find a function `g` such that `g(X) ≈ y`.
  - `X`: Feature matrix (input data, e.g., `[year, make, mileage]`).
  - `y`: Target vector (desired output, e.g., `[price]`).
  - `g`: The model, which is the function we are trying to learn.
- The model `g` is trained to approximate the relationship between the features `X` and the target `y` as closely as possible.
- Once trained, `g` can be used to predict `y` for new, unseen `X`.

---

#### **1.4 CRISP-DM (Cross-Industry Standard Process for Data Mining)**

CRISP-DM is a widely-used, iterative framework for managing data mining projects.
- It consists of six phases arranged in a cycle:
  1. **Business Understanding**: Define the project goals and requirements from a business perspective.
  2. **Data Understanding**: Collect and explore the initial data to understand its characteristics.
  3. **Data Preparation**: Clean, transform, and prepare the data for modeling (e.g., handling missing values, encoding categorical variables).
  4. **Modeling**: Select and apply various modeling techniques (e.g., linear regression, decision trees) to the prepared data.
  5. **Evaluation**: Assess the model's performance against the business objectives to ensure it meets the requirements.
  6. **Deployment**: Put the model into production so it can be used to make real-world predictions.
- The process is iterative; feedback from later stages can lead to revisiting earlier ones (e.g., if evaluation shows poor performance, you might go back to data preparation or modeling).

---

#### **1.5 Model Selection Process**

To select the best model, you must evaluate different models on data they have not seen during training.
- The standard approach is to split your dataset into three parts:
  1. **Training Set**: Used to train the model.
  2. **Validation Set**: Used to tune hyperparameters and compare the performance of different models.
  3. **Test Set**: Used *only once* to get an unbiased final evaluation of the chosen model's performance.
- The steps are:
  1. Split the data into train/validation/test sets.
  2. Train multiple candidate models on the training set.
  3. Evaluate each model on the validation set.
  4. Select the best-performing model based on validation metrics.
  5. Finally, test the selected model on the test set to report its generalization performance.
- This prevents overfitting and ensures the model will perform well on new, unseen data.

---

#### **1.6 Environment Setup**

To follow along with the course, you need to set up a Python environment with key libraries.
- **Required Libraries**: Python, NumPy, Pandas, Matplotlib, Scikit-Learn.
- **Recommended Tool**: Anaconda or Miniconda. Anaconda is a distribution that includes Python and many popular data science packages out-of-the-box, making it the easiest option for beginners.
- **Cloud Option**: You can also use cloud services like AWS or Google Cloud Platform (GCP) for more powerful computing resources. The course provides instructions for setting up an AWS EC2 instance.
- **Notebook Services**: Platforms like Kaggle or Google Colab allow you to run Jupyter notebooks directly in your browser without local setup.

---

#### **1.7 Introduction to NumPy**

NumPy is a fundamental library for scientific computing in Python.
- It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.
- Key functionalities include:
  - Creating arrays (`np.array()`, `np.zeros()`, `np.ones()`).
  - Performing element-wise operations (`+`, `-`, `*`, `/`) on entire arrays.
  - Generating random numbers (`np.random.rand()`, `np.random.randn()`).
  - Linear algebra operations (matrix multiplication, inverse, etc.).
- NumPy forms the foundation for many other data science libraries, including Pandas and Scikit-Learn.

---

#### **1.8 Linear Algebra Refresher**

Linear algebra is essential for understanding how many ML algorithms work under the hood.
- The lecture briefly introduces key concepts:
  - **Vector Operations**: Addition, subtraction, scalar multiplication.
  - **Matrix Operations**: Matrix-vector multiplication, matrix-matrix multiplication.
  - **Identity Matrix**: A square matrix with 1s on the diagonal and 0s elsewhere (`np.eye(n)`).
  - **Inverse Matrix**: The inverse of a matrix `A` is a matrix `A^-1` such that `A @ A^-1 = I`.
- The instructor emphasizes that while formulas may look intimidating, they become manageable when implemented in code (e.g., using NumPy).

---

#### **1.9 Introduction to Pandas**

Pandas is the primary library for data manipulation and analysis in Python.
- It provides two main data structures:
  1. **Series**: A one-dimensional labeled array capable of holding any data type.
  2. **DataFrame**: A two-dimensional, size-mutable, tabular data structure with labeled axes (rows and columns). Think of it as a spreadsheet or SQL table.
- Core functionalities include:
  - Reading data from files (`pd.read_csv()`).
  - Inspecting data (`df.head()`, `df.info()`, `df.shape`).
  - Selecting columns (`df['column_name']`).
  - Vectorized operations (applying functions to entire columns without loops).
- The Index is a crucial component of Pandas, enabling fast data alignment and lookups.

---

#### **Next Steps**

The next session will dive into a hands-on **Car Price Prediction Project**, applying all the concepts learned in this introductory session to build a real-world machine learning model using linear regression.