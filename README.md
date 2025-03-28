# Credit Risk Classification

## 📚 Table of Contents
- [📌 Overview](#-overview)
- [🎯 Objectives](#-objectives)
- [📁 Project Deliverables](#-project-deliverables)
- [🧪 Machine Learning Workflow](#-machine-learning-workflow)
- [📊 Results](#-results)
- [✅ Summary & Recommendation](#-summary--recommendation)
- [🛠️ Technologies Used](#-technologies-used)
- [📖 Steps to Reproduce](#-steps-to-reproduce)
- [📜 Reference Section](#-reference-section)
- [🏆 Acknowledgments](#-acknowledgments)

---

## 📌 Overview

In this challenge, we were tasked to analyze a dataset containing financial details of historical loan recipients to develop a machine learning model capable of predicting credit/loan risks in future applicants. 

The main goal was to create a model that classifies each loan applicant as either a **high-risk applicant (`1`)** or **low-risk applicant (`0`)**, allowing lenders (e.g., banks or private equity institutions) to make informed decisions and reduce the chances of default. 

This model serves as a simplified version of the type of predictive modeling used in real-world credit approval workflows—helping institutions detect and flag potentially risky clients before their loans are issued.

The dataset included several important financial indicators such as `loan_size`, `interest_rate`, `debt_to_income`, and `derogatory_marks`, all of which are commonly used to assess an applicant’s financial health. The target variable `loan_status` was highly imbalanced, with **18,765** loans labeled as low-risk (`0`) and only **619** labeled as high-risk (`1`).

After inspecting the dataset using `.describe()`, I noticed large differences in scale across the features. For example, `borrower_income` had a mean over **49,000**, while `derogatory_marks` ranged only from **0 to 3**. To handle this, I applied **`StandardScaler()`** to normalize the features and prevent variables with large numeric values from disproportionately influencing the logistic regression model.

---

## 🎯 Objectives
- Analyze financial data to assess loan applicant risk
- Normalize the feature data for consistency
- Build a baseline classification model using `LogisticRegression`
- Evaluate model performance with appropriate metrics
- Make a recommendation based on model outcomes

---

## 📁 Project Deliverables

### 📂 File Structure
```
credit-risk-classification/
├── Credit_Risk/
│   ├── credit_risk_classification.ipynb  ← Main analysis notebook
│   └── Resources/
│       └── lending_data.csv              ← Financial dataset
├── README.md                             ← Project summary & documentation
```

---

## 🧪 Machine Learning Workflow

- **Explored the dataset** using `.head()` and `.describe()` to understand data types, distributions, and scaling needs  
- **Separated features and labels**: Assigned `X` for features and `y` for the `loan_status` target  
- **Scaled the X numeric features** using `StandardScaler()` to normalize ranges and center the data  
- **Split the data** into training and testing sets using `train_test_split()` with a fixed `random_state=1`  
- **Trained a `LogisticRegression` model** using the processed training data  
- **Evaluated model performance** using a confusion matrix and classification report, focusing on **accuracy**, **precision**, and **recall**

---

## 📊 Results

After training the logistic regression model, I evaluated its performance using **accuracy**, **precision**, **recall**, and the **confusion matrix**. These metrics offered insight into how well the model distinguishes between low-risk and high-risk applicants.

### 🧠 Model: `LogisticRegression`
- **Accuracy**: **99.36%**  
- **Precision**: **84.07%**  
- **Recall**: **98.39%**  

**Confusion Matrix:**
```
[[18652 113]
 [   10 609]]
```

Out of **619 actual high-risk applicants**, the model correctly identified **609**, missing only **10**—resulting in a strong **recall** score of **98.39%**. It also correctly classified **18,652** out of **18,765** low-risk applicants, producing only **113** false positives.

This high recall indicates that the model is excellent at catching risky applicants. While the **precision** is slightly lower (84.07%), this is an acceptable trade-off in financial settings where missing a high-risk loan could be far more costly than flagging a few low-risk applicants incorrectly.

---

## ✅ Summary & Recommendation

Based on the evaluation results, the logistic regression model performed exceptionally well—especially in terms of **recall**, which reached **98.39%**. This means the model successfully identified nearly all high-risk applicants, which is critical in preventing loan defaults.

The **accuracy** of **99.36%** supports the model’s overall reliability, and the **precision** of **84.07%** confirms that most flagged high-risk cases were valid.

In the context of credit risk, **recall** is more important than precision because the cost of approving a high-risk loan is significantly greater than the cost of rejecting a low-risk one. Therefore, the model’s cautious approach is well-suited to the task.

**✅ Recommendation:** Deploy this logistic regression model as a baseline classifier for credit risk analysis. It handles class imbalance effectively and prioritizes the identification of high-risk applicants.

---

## 🛠️ Technologies Used
- Python 3  
- Pandas  
- Scikit-learn (`LogisticRegression`, `StandardScaler`, metrics)  
- Visual Studio Code (VS Code)  

---

## 📖 Steps to Reproduce

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-classification.git
cd credit-risk-classification
```

2️⃣ **Open in VS Code**
- Launch Visual Studio Code
- Open the `credit-risk-classification` folder

3️⃣ **Run the Notebook**
- Navigate to `Credit_Risk/credit_risk_classification.ipynb`
- Run all cells step-by-step within the integrated Jupyter environment

✅ The notebook includes all preprocessing, training, and evaluation steps required to reproduce the results.

---

## 📜 Reference Section
- Dataset: Provided by edX Boot Camps LLC for educational use  
- All code authored by Sean Schallberger unless otherwise noted

---

## 🏆 Acknowledgments

Project completed by **Sean Schallberger** as part of the University of Toronto Data Analytics Bootcamp.

Special thanks to the instructional team and TAs for their support and feedback throughout the course.

---

> **Challenge:** Module 20 – Credit Risk Classification

