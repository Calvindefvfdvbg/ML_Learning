# Mastering Logistic Regression

Welcome to this comprehensive tutorial on Logistic Regression! If you're diving into machine learning, understanding classification algorithms is crucial. Logistic Regression is a fundamental technique used in supervised learning for binary classification problems—like determining if an email is spam or not, or predicting if a patient has a disease based on symptoms.

## What is Logistic Regression?

Logistic Regression is a supervised learning algorithm primarily used for **classification tasks**. Unlike regression models that predict continuous values (e.g., house prices), it predicts categories or classes. The simplest form is binary classification, where the output is either "yes" or "no," "0" or "1," "pass" or "fail."

Common examples:
- Email classification: Spam or Not Spam?
- Medical diagnosis: Does the patient have a disease (Yes/No)?
- Customer churn: Will the customer leave (Yes/No)?

In essence, it helps decide which category an input belongs to based on input features.

<img width="1536" height="1024" alt="Segmod function " src="https://github.com/user-attachments/assets/56c4aa0d-b3b3-4f94-a6ae-df3ed9b92752" />



## How It Differs from Linear Regression

If you've studied Linear Regression (where you predict continuous values like sales based on advertising spend), Logistic Regression builds on similar ideas but adapts them for classification.
<img width="1536" height="1024" alt="img1" src="https://github.com/user-attachments/assets/b28f0a65-1e35-4cd6-bb07-ef6340d3056b" />



Key point: The dependent variable we're predicting is discrete (categories), not continuous.

## The Sigmoid Function: The Heart of Logistic Regression

The magic happens with the Sigmoid Function, which transforms any real-valued number into a value between 0 and 1. This represents the probability of belonging to the positive class (e.g., "pass" = 1).

<img width="1536" height="1024" alt="Sigmoid Formula" src="https://github.com/user-attachments/assets/b2d4d989-53d4-4679-9998-60670270a694" />



Output interpretation:
- If σ(z) > 0.5, classify as 1 (e.g., pass).
- If σ(z) ≤ 0.5, classify as 0 (e.g., fail).
- The value itself is the probability (e.g., 0.8 means 80% chance of class 1).

This creates an S-shaped curve, dividing data into two classes.

## Real-World Example: Predicting Exam Results Based on Study Hours

Let's use a simple example from the transcript: Predicting if a student will pass (1) or fail (0) an exam based on study hours (independent variable).

<img width="1024" height="1536" alt="Traning_data" src="https://github.com/user-attachments/assets/611cefd2-8546-4933-bdbb-947675735857" />


This shows how the model calculates probabilities and classifies based on a threshold (usually 0.5).

## How Are Coefficients (a0, a1) Calculated?

The model learns \( a_0 \) and \( a_1 \) using **Maximum Likelihood Estimation (MLE)**. This method finds the parameters that make the observed data most probable.

To optimize:
- Use a **cost function** (like log-loss) to measure errors.
- Minimize the cost with **Gradient Descent** (iteratively adjusting parameters) or specialized solvers.

This involves math like derivatives, but libraries handle it for you. In exams or interviews, they might give you pre-calculated values or ask for conceptual understanding.

## Machine Learning Example: Implementing Logistic Regression in Python

Let's put theory into practice! We'll use Python's scikit-learn library on the Iris dataset (classifying flowers as Setosa or not). This is a binary classification setup.

First, install dependencies if needed (but assume you have them: numpy, pandas, scikit-learn).

Here's the code:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# For binary classification: Setosa (0) vs Non-Setosa (1)
df['target'] = np.where(df['target'] == 0, 0, 1)  # 0: Setosa, 1: Others

# Features (independent variables): Petal length and width
X = df[['petal length (cm)', 'petal width (cm)']]
y = df['target']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Coefficients
print(f"Intercept (a0): {model.intercept_[0]}")
print(f"Coefficients (a1 for each feature): {model.coef_[0]}")

# Example prediction: New flower with petal length 1.4 cm, width 0.2 cm
new_flower = np.array([[1.4, 0.2]])
pred_prob = model.predict_proba(new_flower)[0][1]  # Probability of class 1
pred_class = model.predict(new_flower)[0]
print(f"Predicted Probability (Non-Setosa): {pred_prob:.4f}")
print(f"Predicted Class: {pred_class} (0=Setosa, 1=Non-Setosa)")
```

### Explanation of the Code:
- **Data Loading**: We use the Iris dataset, converting it to binary (Setosa vs others).
- **Features**: Petal length and width as inputs.
- **Training**: Fit the LogisticRegression model.
- **Prediction**: Get probabilities and classes.
- **Evaluation**: Accuracy and confusion matrix show performance.
- **New Prediction**: For a sample input, it outputs probability (using sigmoid internally) and class.

Sample Output (results may vary slightly):
- Accuracy: 100.00%
- Confusion Matrix: [[10  0] [ 0 20]]
- Intercept: -5.83 (approx.)
- Coefficients: [2.20, 1.45] (approx.)
- For new flower: Probability ~0.001 (very low, classifies as Setosa=0).

This demonstrates how Logistic Regression classifies with probabilities.

---------



# Logistic Regression Tutorial with a Real Student Dataset

In this tutorial, we'll dive deeper into Logistic Regression by creating a **realistic dataset** simulating student exam performance. We'll use this dataset to predict whether a student passes (1) or fails (0) an exam based on features like study hours, sleep hours, and prior test scores. This builds on the concepts from the previous Logistic Regression tutorial, inspired by the provided video transcript, and includes a hands-on Python example using scikit-learn.

<img width="1024" height="1536" alt="ChatGPT Image Aug 21, 2025, 03_17_36 PM" src="https://github.com/user-attachments/assets/d4368abb-926d-42ea-837c-ab3b6b483327" />


## Creating a Realistic Student Dataset

To make this practical, we’ll create a synthetic dataset that mimics real-world student data. The dataset includes:
- **Study Hours**: Hours spent studying per week (continuous).
- **Sleep Hours**: Average sleep hours per night (continuous).
- **Prior Test Score**: Score on a previous test (0–100, continuous).
- **Pass**: Target variable (0 = Fail, 1 = Pass, binary).

### Dataset Creation
We’ll generate a dataset with 100 student records. The data is designed to reflect realistic patterns (e.g., more study hours and higher prior scores increase pass likelihood).

```python
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate 100 student records
n_samples = 100
study_hours = np.random.normal(5, 2, n_samples).clip(0, 10)  # Mean 5, std 2, clipped 0-10
sleep_hours = np.random.normal(7, 1, n_samples).clip(4, 10)  # Mean 7, std 1, clipped 4-10
prior_test_score = np.random.normal(70, 15, n_samples).clip(0, 100)  # Mean 70, std 15

# Simulate pass/fail (1/0) based on a simple rule with noise
# Higher study hours, sleep, and prior scores increase pass probability
logit = -5 + 0.5 * study_hours + 0.3 * sleep_hours + 0.05 * prior_test_score
probs = 1 / (1 + np.exp(-logit))  # Sigmoid to get probabilities
pass_exam = (probs > 0.5 + np.random.normal(0, 0.1, n_samples)).astype(int)

# Create DataFrame
data = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'prior_test_score': prior_test_score,
    'pass_exam': pass_exam
})

# Save dataset to CSV (optional, for reference)
data.to_csv('student_exam_data.csv', index=False)

# Display first few rows
print(data.head())
```

### Dataset Explanation
- **Columns**:
  - `study_hours`: Continuous (0–10 hours), normally distributed.
  - `sleep_hours`: Continuous (4–10 hours), normally distributed.
  - `prior_test_score`: Continuous (0–100), normally distributed.
  - `pass_exam`: Binary (0 = Fail, 1 = Pass).
- **Logic**: The `pass_exam` variable is generated using a logistic-like rule where higher values of study hours, sleep hours, and prior test scores increase the probability of passing. Noise is added to simulate real-world variability.
- **Size**: 100 records, sufficient for a simple Logistic Regression model.

Sample output of `data.head()`:
```
   study_hours  sleep_hours  prior_test_score  pass_exam
0     5.493428     6.882026         76.943484          1
1     4.723471     6.038772         58.725056          0
2     6.295015     7.285351         68.697534          1
3     8.046060     6.642234         87.654623          1
4     4.531693     7.933796         64.485739          0
```

## Logistic Regression Implementation in Python

Now, let’s use this dataset to train a Logistic Regression model to predict whether a student passes or fails.

### Step-by-Step Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (assuming it's saved from above)
data = pd.read_csv('student_exam_data.csv')

# Features (independent variables) and target
X = data[['study_hours', 'sleep_hours', 'prior_test_score']]
y = data['pass_exam']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print(f"Intercept (a0): {model.intercept_[0]:.4f}")
print(f"Coefficients (a1 for each feature): {model.coef_[0]}")

# Example prediction for a new student
new_student = np.array([[5, 7, 80]])  # 5 study hours, 7 sleep hours, 80 prior score
pred_prob = model.predict_proba(new_student)[0][1]  # Probability of passing
pred_class = model.predict(new_student)[0]
print(f"\nNew Student (5 study hrs, 7 sleep hrs, 80 prior score):")
print(f"Predicted Probability of Passing: {pred_prob:.4f}")
print(f"Predicted Class: {pred_class} (0=Fail, 1=Pass)")

# Visualize the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

### Code Explanation
1. **Dataset Loading**: We load the generated dataset (or recreate it).
2. **Features and Target**:
   - Features: `study_hours`, `sleep_hours`, `prior_test_score`.
   - Target: `pass_exam` (0 or 1).
3. **Train-Test Split**: 80% training, 20% testing.
4. **Model Training**: LogisticRegression from scikit-learn, with `max_iter=200` to ensure convergence.
5. **Evaluation**:
   - **Accuracy**: Percentage of correct predictions.
   - **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives.
6. **Coefficients**: The model’s intercept (\(a_0\)) and coefficients (\(a_1\)) for each feature.
7. **Prediction**: For a new student (5 study hours, 7 sleep hours, 80 prior score), we predict the probability and class.
8. **Visualization**: A heatmap of the confusion matrix for better understanding.

### Sample Output
(Results may vary slightly due to randomness in the dataset):
```
Accuracy: 85.00%
Confusion Matrix:
[[8 2]
 [1 9]]
Intercept (a0): -6.1234
Coefficients (a1 for each feature): [0.4512 0.2876 0.0489]

New Student (5 study hrs, 7 sleep hrs, 80 prior score):
Predicted Probability of Passing: 0.7823
Predicted Class: 1 (0=Fail, 1=Pass)
```

The confusion matrix plot shows:
- True Negatives (top-left): Correctly predicted Fail.
- True Positives (bottom-right): Correctly predicted Pass.
- False Positives/Negatives: Errors in prediction.

## Understanding the Results
- **Accuracy (~85%)**: The model correctly predicts pass/fail for 85% of test cases.
- **Coefficients**: Positive coefficients indicate that higher study hours, sleep hours, and prior test scores increase the likelihood of passing.
- **New Student Prediction**: With 5 study hours, 7 sleep hours, and an 80 prior score, the model predicts a 78.23% chance of passing, classifying as Pass (1).

## Key Takeaways
- Logistic Regression is ideal for binary classification tasks like pass/fail prediction.
- The **Sigmoid Function** converts a linear combination of features into probabilities (0–1).
- The model learns weights (\(a_0\), \(a_1\)) via Maximum Likelihood Estimation, optimized with Gradient Descent.
- Real-world datasets require careful feature selection and preprocessing for better performance.

## Next Steps
- **Experiment**: Try adding features like attendance or study method (categorical).
- **Tune**: Adjust the decision threshold (default 0.5) or try regularization (e.g., `C` in LogisticRegression).
- **Explore**: Use real datasets from platforms like Kaggle to practice.







