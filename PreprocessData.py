def remove_missing_values(dataset):
    dataset = dataset.drop(dataset[dataset['TotalCharges'] == " "].index)
    return dataset