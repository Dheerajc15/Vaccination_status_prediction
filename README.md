# Vaccination Status Prediction
**Project Overview:**
Vaccination remains a critical public health measure, particularly during pandemics. Understanding the factors that influence vaccine uptake can guide interventions and awareness campaigns. This project aimed to predict whether an individual is vaccinated based on demographic, health, and behavioral data. By leveraging machine learning models, the project provides actionable insights for public health planning and policy-making.

**Dataset Description:**
The dataset, "h1n1_vaccine_prediction.csv," includes data points capturing diverse features such as:

- **Demographics:** Age, gender, income levels, and education.
- **Health Information:** Pre-existing conditions, flu symptoms, and overall health status.
- **Behavioral Factors:** Vaccine awareness, access to healthcare, and vaccine hesitancy levels.

**Problem Statement:**
To predict whether an individual has received the H1N1 vaccine based on the provided features. This binary classification task involved distinguishing between vaccinated and unvaccinated individuals.

**Methodology:**
The project adopted a systematic approach to data preprocessing, feature engineering, model development, and evaluation:

1. **Data Cleaning and Preprocessing:**
   - Addressed missing values through imputation techniques.
   - Normalized numerical features and encoded categorical variables using one-hot and label encoding.
   - Conducted exploratory data analysis (EDA) to identify trends and correlations.

2. **Exploratory Data Analysis (EDA):**
   - Visualized vaccination rates across demographic groups using Seaborn and Matplotlib.
   - Examined correlations between vaccine uptake and features like vaccine awareness and health conditions.
   - Identified key factors contributing to vaccine hesitancy.

3. **Feature Engineering:**
   - Created derived features, such as risk indices based on health conditions.
   - Combined related categorical features to reduce dimensionality.

4. **Model Building and Evaluation:**
   - Trained several machine learning models, including:
     - **Logistic Regression:** A baseline model to assess linear relationships.
     - **Random Forest:** To capture complex feature interactions.
     - **Gradient Boosting:** For enhanced performance on imbalanced data.
     - **AdaBoost:** To reduce bias and variance.
     - **K-Nearest Neighbors (KNN):** For simplicity and interpretability.
     - **Decision Tree Classifier:** To visualize decision boundaries.
   - Evaluated models using:
     - **Accuracy:** To measure overall correctness.
     - **F1-Score:** To balance precision and recall in an imbalanced dataset.
     - **Confusion Matrix:** To analyze true positives, false positives, true negatives, and false negatives.
     - **Cross-Validation:** To ensure model robustness and avoid overfitting.

5. **Hyperparameter Tuning:**
   - Conducted grid search to optimize model parameters for Random Forest, Gradient Boosting, and Logistic Regression.

**Key Results:**
- The **Random Forest Classifier** achieved the highest accuracy of over **90%**, with strong F1-scores and precision.
- Logistic Regression served as a reliable baseline with interpretable results, achieving approximately **85%** accuracy.
- Feature importance analysis highlighted that factors like vaccine awareness and access to healthcare were the strongest predictors of vaccine uptake.

**Challenges and Solutions:**
- **Imbalanced Dataset:** The majority class dominated the dataset. Techniques like SMOTE (Synthetic Minority Over-sampling Technique) were used to balance the target classes.
- **Multicollinearity:** High correlations among some features were managed using variance inflation factors (VIF) and feature selection.

**Impact:**
This project underscores the potential of machine learning in public health. By accurately predicting vaccine uptake, it provides:

- Insights into key drivers of vaccine hesitancy and acceptance.
- Tools for targeting awareness campaigns to specific groups.
- Support for policymakers in allocating resources for vaccination programs.

**Tools and Technologies Used:**
- **Programming Languages:** Python (pandas, NumPy, scikit-learn, Matplotlib, Seaborn)
- **Libraries for Modeling:** scikit-learn (Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, etc.)
- **Visualization Tools:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

**Future Scope:**
- Expand the dataset to include environmental and social factors influencing vaccine uptake.
- Explore deep learning techniques to capture non-linear patterns in large-scale data.
- Collaborate with healthcare providers to validate predictions in real-world settings.

**Conclusion:**
By leveraging machine learning, this project highlights the importance of data-driven approaches in tackling public health challenges. The predictive models developed here can assist in identifying at-risk populations, informing intervention strategies, and ultimately improving vaccination rates.

