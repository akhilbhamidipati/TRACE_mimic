import os

def write_patient_data():
    """ loop through patient folders and get label for each one

    Returns:
        dict: dictionary of patients and their labels
    """
    patient_labels = {}
    valid_patients = ['211', '471', '476', '449', '041', '413', '414', '221', '226', '415', '401', '408', '260', '409', '055', '037', '039', '248', '417', '212', '472', '240', '418', '427', '474', '442', '213', '231', '253', '254', '291', '237', '230', '466', '403', '252']
   
    valid_patients_labels = dict()
    first_cardiac_arrest_records = dict()
    
    with open('labels.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            patient_id, label = line.split(",")
            label = int(label)
            
            if patient_id in valid_patients:
                valid_patients_labels[patient_id] = label
    
    with open('first_cardiac_arrest_record.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            patient_id, index = line.split(",")
            index = int(index)
            
            if patient_id in valid_patients:
                first_cardiac_arrest_records[patient_id] = index
    

    for folder_name in os.listdir(os.getcwd()):
        if folder_name.isdigit() and folder_name in valid_patients:
            print(f"Getting data for patient name: {folder_name}")
            # loop through every file
            patient_data = get_data_per_patient(folder_name)
            write_data_to_file(folder_name, patient_data, valid_patients_labels, first_cardiac_arrest_records)
            
            
    return patient_labels

def get_data_per_patient(folder_name):
    """Gets data for each patient in terms of columns

    Args:
        folder_name (str): patient name
    Returns:
        dict<str, list>: dictionary signal names as keys and a list of values for each signal in order
    """
    patient_data = {}
    file_names = os.listdir(os.path.join(os.getcwd(), folder_name))
    for file_name in file_names:
        # if the file name ends with txt
        if file_name.endswith((".txt")):
            file_path = os.path.join(os.getcwd(), folder_name, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith(("SpO2")) or line.startswith(("RESP")) or line.startswith(("HR")) or line.startswith(("ABP")):
                        reading = line.split("\t")
                        if len(reading) == 2:
                            signal = reading[0].strip()
                            value = int(float(reading[1]))
                            #print(f"Signal: {signal}, Value: {value}")
                            if signal != "ABP":
                                if signal not in patient_data:
                                    patient_data[signal] = [value]
                                else:
                                    patient_data[signal].append(value)
                            else:
                                if "ABP1" not in patient_data and "ABP2" not in patient_data and "ABP3" not in patient_data:
                                    patient_data["ABP1"] = [value]
                                    patient_data["ABP2"] = [value]
                                    patient_data["ABP3"] = [value]
                                else:
                                    patient_data["ABP1"].append(value)
                                    patient_data["ABP2"].append(value)
                                    patient_data["ABP3"].append(value)
                        elif len(reading) > 2:
                            values = reading[1:]
                            for index, value in enumerate(values):
                                currentABP = f"ABP{index+1}"
                                #print(f"Signal: {currentABP}, Value: {value}")
                                if currentABP not in patient_data:
                                    patient_data[currentABP] = [int(value)]
                                else:
                                    patient_data[currentABP].append(int(value))
    return patient_data

def write_data_to_file(patient_id, patient_data, valid_patients_labels, first_cardiac_arrest_records):
    # find which patients are positive and negative
    NUM_RECORDS = 10000

    with open(f'valid_patient_data/{patient_id}_data.csv', 'w', encoding='utf-8') as f:
        temp_record = {}
        record_number = None
        if valid_patients_labels[patient_id] == 0:
            record_number = NUM_RECORDS
        else:
            record_number = max(NUM_RECORDS, first_cardiac_arrest_records[patient_id])
        
        # print(record_number-NUM_RECORDS)
        # print(record_number)
        signal_list = ["SpO2", "RESP", "HR", "ABP1", "ABP2", "ABP3"]
        f.write("record_id,SpO2,RESP,HR,ABP1,ABP2,ABP3\n")

        for record_id in range(record_number-NUM_RECORDS,  record_number):
            for signal in signal_list:
                temp_record[signal] = patient_data[signal][record_id]
            f.write("{},{},{},{},{},{},{}\n".format(record_id+1, temp_record["SpO2"], temp_record["RESP"], temp_record["HR"], temp_record["ABP1"], temp_record["ABP2"], temp_record["ABP3"]))

def main():
    write_patient_data()
    print("done!")

if __name__ == "__main__":
    main()
