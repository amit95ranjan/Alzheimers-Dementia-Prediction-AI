# 🧠 Dementia Patient Health Analysis & Prediction

## 📌 Project Overview
This project analyzes a dataset of dementia patients to identify key health indicators, lifestyle factors, and medical history variables associated with the diagnosis. By leveraging Python for data manipulation and Scikit-Learn for predictive modeling, the project aims to determine the most significant risk factors and build a classification model to predict dementia onset.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Random Forest, Logistic Regression)

## 📂 Data Description
The dataset includes patient health records with features such as:
* **Demographics:** Age, Family History
* **Vitals & Labs:** Dosage in mg, APOE_ε4 (Genetic marker)
* **Lifestyle:** Smoking Status, Sleep Quality, Alcohol Level
* **Medical History:** Chronic Health Conditions, Depression Status, Diabetes, Prescriptions

## 📊 Key Workflow & Methodology

### 1. Data Preprocessing & Cleaning
* **Handling Missing Values:**
    * *Chronic Health Conditions* → Imputed with 'Unknown' to preserve data rows.
    * *Prescription* → Imputed with 'Not Prescribed'.
    * *Dosage* → Imputed with 0 for non-medicated patients.
* **Feature Engineering:** Separation of categorical and numerical features for targeted analysis.

### 2. Exploratory Data Analysis (EDA)
Comprehensive visualization was performed to understand data distribution and correlations:
* **Distribution Analysis:** Histograms and KDE plots for numerical variables.
* **Categorical Breakdowns:** Bar charts for prescriptions and chronic conditions.
* **Correlation Heatmap:** Identified relationships between numerical features.
* **Multivariate Analysis:**
    * *Violin Plots:* Analyzed dosage distribution across different prescriptions.
    * *Stacked Bar Charts:* Compared Smoking Habits vs. Chronic Diseases and Sleep Quality impact.

### 3. Key Insights
* **Depression & Dementia:** A strong correlation was observed where patients diagnosed with Dementia frequently exhibited status of Depression.
* **Sleep Quality:** Poor sleep quality showed a notable relationship with chronic health conditions.
* **Feature Importance:** Using a Random Forest Classifier, the top 5 most dependable variables for diagnosis were identified (including Genetic markers and Age).

### 4. Machine Learning Models
Two classification models were implemented to predict Dementia status:
1.  **Random Forest Classifier:** Used for its ability to handle non-linear relationships and provide feature importance.
2.  **Logistic Regression:** Used as a baseline model for binary classification.

**Model Evaluation:**
* Accuracy Score
* Cross-Validation (K-Fold)
* Confusion Matrix & Classification Report (Precision, Recall, F1-Score)

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone 
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Dementia_Analysis.ipynb
    ```

## 🔮 Future Scope
* **Class Imbalance Handling:** Implement SMOTE (Synthetic Minority Over-sampling Technique) to improve recall on the positive class.
* **Hyperparameter Tuning:** Use GridSearchCV to optimize the Random Forest parameters.
* **Deep Learning:** Experiment with Neural Networks for potential accuracy improvements.

---
*Created by Amit Ranjan - Data Analyst*
