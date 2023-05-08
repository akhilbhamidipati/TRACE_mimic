import os
import torch

# Number of records to observe
NUM_RECORDS = 10000
TEST_HOLDOUT = 6

"""
All patients did not have the same set of physiological signals recorded. Therefore, we selected a subset of the
patients for whom SpO2, repspiratory rate, heart rate, ABP, and Pulse were all measured and for whom the recordings
span at least 3 hours.
"""
VALID_PATIENTS = ['211', '471', '476', '449', '041', '413', '414', '221', '226', '415', '401', '408', '260', '409', '055', '037', '039', '248', '417', '212', '472', '240', '418', '427', '474', '442', '213', '231', '253', '254', '291', '237', '230', '466', '403', '252']

def load_labels():
    # valid patients are patients who have all of the following physiological signals recorded: "SpO2", "RESP", "HR", "ABP", "PULSE"
    valid_patients_labels = dict()
    
    with open('mimic-database/labels.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            patient_id, label = line.split(",")
            label = int(label)
            
            if patient_id in VALID_PATIENTS:
                valid_patients_labels[patient_id] = label
    
    labels = []
    for patient_id in sorted(valid_patients_labels.keys()):
        labels.append(valid_patients_labels[patient_id])
    
    train_labels = labels[:-TEST_HOLDOUT]
    test_labels = labels[-TEST_HOLDOUT:]
    
    return torch.tensor(train_labels), torch.tensor(test_labels)

def load_data():
    path = 'mimic-database/valid_patient_data'
    file_names = os.listdir(os.path.join(os.getcwd(), path))
    
    signal_list = ["SpO2", "RESP", "HR", "ABP1", "ABP2", "ABP3"]
    data = torch.zeros((len(file_names), len(signal_list), NUM_RECORDS))
    
    patient_idx = 0
    
    for file_name in sorted(file_names):
        this_patient_data = torch.zeros((len(signal_list), NUM_RECORDS))
        index = 0
        with open(f"{path}/{file_name}", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:
                records = [int(x) for x in line.split(",")]
                for j in range(len(signal_list)):
                    this_patient_data[j][index] = records[j+1]
                index += 1
        
        data[patient_idx] = this_patient_data
        patient_idx += 1
    
    train_data, test_data = data[:-TEST_HOLDOUT], data[-TEST_HOLDOUT:]
    return train_data, test_data