import pandas as pd
from AnalysisData import plot_churn_pie, plot_gender_histogram, plot_gender_pie, plot_monthly_total_charges, plot_contract_churn_relationship,plot_online_security_analysis, plot_services_pie
from PreprocessData import remove_missing_values, convert_data_types
# Read the dataset
Dataset=pd.read_csv("/content/drive/MyDrive/AI_Developer/WA_Fn-UseC_-Telco-Customer-Churn.csv")
Dataset.drop('customerID', axis=1, inplace=True)

# clean-data
cleaned_dataset = remove_missing_values(Dataset)
cleaned_dataset = convert_data_types(cleaned_dataset)

# Call the plotting functions
plot_churn_pie(cleaned_dataset)
plot_gender_histogram(cleaned_dataset)
plot_gender_pie(cleaned_dataset)
plot_monthly_total_charges(cleaned_dataset)
plot_contract_churn_relationship(cleaned_dataset)
plot_online_security_analysis(cleaned_dataset)
plot_services_pie(cleaned_dataset)