# Module 20 Report Template

## 📌 Overview of the Analysis

In this challenge, we were tasked to analyze a dataset containing financial details of historical loan recipients to develop a machine learning model capable of predicting credit/ loan risks in future applicants. Essentially, the main goal is create an ML model which can classify each loan applicant as either a **high-risk applicant (`1`)** or **low-risk applicant (`0`)** , allowing lenders (banks, private equity institutions etc.) to make informed decisions to minimize default risk. This model is a largely simplified form of the type of predictive modeling that plays a critical role in credit approval and management workflows allowing institutions to identify and further analyze potentially risky clients before their loans are issued or become defaulted.

The dataset provided included several important financial indicators such as `loan_size`, `interest_rate`, `debt_to_income`, and `derogatory_marks`, which are often used to evaluate a borrower’s financial health. Considering these metrics, it was decided to make use of `loan_status` as the target variable producing an imbalanced result in which **18,765** loans were classified as low-risk (`0`), while only **619** were labeled high-risk (`1`).

Additionally upon inspecting the dataset using `.describe()`, I actually happened to also notice that the features across the board had very different scales (mean/ standard deviation). For instance, `borrower_income` had a mean of over **49,000** and a standard deviation above **8,000**, while `derogatory_marks` ranged from **0 to 3**. To address this, I added a scaling step using **`StandardScaler`** to normalize the numerical features within X leaving the target y values as is. Standardizing the data essentially helped us prevent data or noise from certain features from dominating the ML model due to their larger numeric values. 

### 🧪 Machine Learning Workflow

- **Explored the dataset** using `.head()` and `.describe()` to understand data types, distributions, and scaling needs  
- **Separated features and labels**: Assigned `X` for features and `y` for the `loan_status` target  
- **Scaled the X numeric features** using `StandardScaler` to normalize ranges and center the data  
- **Split the data** into training and testing sets using `train_test_split` with a fixed `random_state` for reproducibility  
- **Trained a `LogisticRegression` model** using the processed training data  
- **Evaluated model performance** using a confusion matrix and classification report, focusing on **accuracy**, **precision**, and **recall**

This process helped build a strong baseline classification model and assess how well logistic regression could identify both low-risk and high-risk applicants based on borrower financial data.


## 📊 Results

After training the logistic regression model, I evaluated its performance using the **accuracy**, **precision**, **recall**, and **confusion matrix** functions. These metrics helped provide insight into how well the model distinguishes between low-risk and high-risk applicants and how reliable its predictions are in a real-world predicting contexts.

### 🧠 Model: `LogisticRegression` Results

- **Accuracy**: **99.36%**  
- **Precision**: **84.07%**  
- **Recall**: **98.39%**

**Confusion Matrix:**
[[18652 113],
[ 10 609]]


Looking at the results of the ML model, out of **619 actual defaulted applicants**, the model correctly identified **609** as high-risk, missing only **10**—resulting in a strong **recall** score of **98.39%**. This suggests that the model is highly effective at catching risky cases. It also correctly predicted **18,652** out of **18,765** low-risk applicants, with only **113** false positives.  

The overall **accuracy** of **99.36%** shows that the model performs consistently across both classes, while the **precision** of **84.07%** indicates that most loan applicants predicted as high-risk were indeed risky. Though a small number of low-risk applicants were over-flagged, this trade-off is acceptable in a financial setting where it's better to be cautious than to overlook potential defaults.


## ✅ Summary & Recommendation

Based on the evaluation results, the logistic regression model performed exceptionally well—especially in terms of recall, which reached **98.39%**. This means the model successfully identified nearly all high-risk applicants, which is critical in preventing loan defaults. The **accuracy** of **99.36%** further supports the model’s overall reliability, and the **precision** of **84.07%** shows that most flagged high-risk cases were indeed valid.

In the context of credit risk, it is expected that recall would be considered to be more important than precision, as missing a high-risk applicant (false negative) can have more serious consequences than incorrectly flagging a low-risk one. Given this, the model’s conservative approach is well-suited for the task.

**Recommendation:**  
This logistic regression model is recommended as a solid baseline for credit risk prediction. It handles class imbalance well and prioritizes catching high-risk cases.

