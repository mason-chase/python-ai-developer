
# Telco Customer Churn

---


```
Please note that the order of codes, files and folders in this project has changed over time. Here are the steps I followed:
*   Searching,
*   Coding,
*   Analyzing the results and refining the approach,
Through repeated iterations, I have learned valuable lessons and adjusted the project structure accordingly. Despite the changes, I believe that my report will be well-suited to the current structure. Additionally, this is the first time I have written tests for my code.
```






---

##  **Classification problem**


The task involved binary classification with the target variable Churn. I began by analyzing the data, and then performed some data cleaning.

Initially, I noticed that the dataset was unbalanced, with a larger number of rows labeled as 'No' for Churn compared to 'Yes'. As a result, I experimented with various ensemble methods later on.

One of the steps in the data cleaning process involves checking the correctness of data types and filling in null values in the dataset. Additionally, since we have a large subset of categorical features, I converted them into numerical features using one-hot-encoding.

In the Exploratory Data Analysis (EDA) step, I visualized the data from various perspectives, such as histograms, scatterplots, and box plots, to gain insights into the data distribution, identify any outliers, and detect any potential correlations between variables.

In the final step, I trained several classification models and analyzed their results to select the best one. For each model, I evaluated its performance on both the training and validation sets to detect any bias or variance issues. After comparing the results of all models, I chose the one that I believed had the best trade-off between bias and variance.d variance.

In this notebook, I provide a step-by-step explanation of what I did in the project. However, in the project folder, I also wrote some test functions using Pytest. Since this is my first time writing tests, I believe it would be beneficial to add more functions to test different parts of the code.



# Please read report.ipyn file or report.html