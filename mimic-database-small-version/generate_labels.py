import os

def get_patient_labels():
    """ loop through patient folders and get label for each one

    Returns:
        dict: dictionary of patients and their labels
    """
    patient_labels = {}
    first_cardiac_arrest_records = {}

    for folder_name in os.listdir(os.getcwd()):
        if folder_name.isdigit():
            print(f"Getting label for patient name: {folder_name}")
            # loop through every file
            had_cardiac_arrest, first_cardiac_arrest_record = get_had_cardiac_arrest(folder_name)
            if had_cardiac_arrest:
                patient_labels[folder_name] = 1
                first_cardiac_arrest_records[folder_name] = first_cardiac_arrest_record
            else:
                patient_labels[folder_name] = 0
    return patient_labels, first_cardiac_arrest_records


def get_had_cardiac_arrest(folder_name):
    """ gets cardiac arrest boolean from files in patient folder

    Args:
        folderName (str): the name of the folder

    Returns:
        boolean: whether the patient had a cardiac arrest event
    """
    file_names = os.listdir(os.path.join(os.getcwd(), folder_name))
    pulse_record_ct = 0
    for file_name in file_names:
        # if the file name ends with txt
        if file_name.endswith((".txt")):
            file_path = os.path.join(os.getcwd(), folder_name, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith(("PULSE")):
                        pulse_record_ct += 1
                        reading = line.split("\t")
                        if len(reading) >= 2:
                            pulse = int(reading[1])
                            if pulse == 0:
                                return True, pulse_record_ct
    return False, None

def write_labels_to_file(patient_labels, first_cardiac_arrest_records):
    with open('labels.csv', 'w', encoding='utf-8') as f:
        for patient, label in patient_labels.items():
            f.write(f"{patient},{label}\n")

    with open('first_cardiac_arrest_record.csv', 'w', encoding='utf-8') as f:
        for patient, index in first_cardiac_arrest_records.items():
            f.write(f"{patient},{index}\n")

def main():
    patient_labels, first_cardiac_arrest_records = get_patient_labels()
    write_labels_to_file(patient_labels, first_cardiac_arrest_records)
    print("done!")

if __name__ == "__main__":
    main()
