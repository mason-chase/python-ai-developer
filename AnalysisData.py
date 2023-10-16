import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
from matplotlib.patches import ConnectionPatch
import numpy as np

def plot_churn_pie(dataset):
    churn_counts = dataset['Churn'].value_counts()
    churn_percentages = churn_counts * 100.0 / len(dataset)
    explode = [0, 0.05]
    churn_percentages.plot.pie(autopct='%.2f%%', explode=explode)
    plt.title('Churn Analysis')
    plt.ylabel('')
    plt.show()

def plot_gender_histogram(dataset):
    plt.hist(dataset['gender'])
    plt.title('Gender Analysis')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()

def plot_gender_pie(dataset):
    gender_counts = dataset['gender'].value_counts()
    gender_percentages = gender_counts * 100.0 / len(dataset)
    explode = [0, 0.03]
    gender_percentages.plot.pie(autopct='%.2f%%', explode=explode)
    plt.title('Gender Analysis')
    plt.ylabel('')
    plt.show()

def plot_monthly_total_charges(dataset):
    sea.scatterplot(x='MonthlyCharges', y='TotalCharges', data=dataset, hue="Churn")
    plt.xlabel('Monthly Charges')
    plt.ylabel('Total Charges')
    plt.title('Relationship between MonthlyCharges, TotalCharges, and Churn')
    plt.show()

def plot_contract_churn_relationship(dataset):
    sea.countplot(x='Contract', hue='Churn', data=dataset)
    plt.title('Contract vs Churn')
    plt.xlabel('Contract')
    plt.ylabel('Number of Customers')
    plt.show()

def plot_online_security_analysis(dataset):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    wedges, *_ = ax1.pie(dataset['OnlineSecurity'].value_counts() * 100.0 / len(dataset),
                         autopct='%.2f%%', explode=[0, 0, 0.05], startangle=35,
                         labels=dataset['OnlineSecurity'].unique())
    theta1, theta2 = wedges[2].theta1, wedges[2].theta2
    center, r = wedges[2].center, wedges[2].r

    age_ratios = dataset[dataset['OnlineSecurity'] == 'No internet service']['SeniorCitizen'].value_counts() / \
                 len(dataset[dataset['OnlineSecurity'] == 'No internet service'])
    age_labels = ['Young', 'Senior']
    bottom = 1
    width = 0.2

    for j, (height, label) in enumerate(reversed([*zip(age_ratios, age_labels)])):
        bottom -= height
        bc = ax2.bar(0, height, width, bottom=bottom, color='C02', label=label, alpha=0.1 + 0.25 * j)
        ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

    ax2.set_title('SeniorCitizen')
    ax2.legend()
    ax2.axis('off')
    ax2.set_xlim(-2.5 * width, 2.5 * width)

    x = r * np.cos(np.pi / 180 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    con.set_linewidth(4)
    ax2.add_artist(con)

    x = r * np.cos(np.pi / 180 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                          xyB=(x, y), coordsB=ax1.transData)
    con.set_color([0, 0, 0])
    ax2.add_artist(con)
    con.set_linewidth(4)


def plot_services_pie(dataset):
    services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    service_counts = dataset[services].apply(pd.value_counts)
    service_counts.plot(kind='pie', subplots=True, figsize=(20, 15), layout=(3, 3), autopct='%.1f%%', colors=['orange', 'purple', 'gray', 'lightblue'])
    plt.show()
