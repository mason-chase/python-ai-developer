def remove_missing_values(dataset):
    dataset = dataset.drop(dataset[dataset['TotalCharges'] == " "].index)
    return dataset

def convert_data_types(dataset):
    dataset['TotalCharges'] = dataset['TotalCharges'].astype(float)
    dataset['SeniorCitizen'] = dataset['SeniorCitizen'].astype(object)
    return dataset