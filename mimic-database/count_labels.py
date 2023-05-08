def main():
    with open('labels.csv', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        count_0 = 0
        count_1 = 0
        for line in lines:
            patient_id, label = line.split(",")
            patient_id = int(patient_id)
            label = int(label)
            if label == 0:
                count_0 += 1
            else:
                count_1 += 1
        print(f"Number of patients with cardiac arrest: {count_0}")
        print(f"Number of patients with no cardiac arrest: {count_1}")   

if __name__ == "__main__":
    main()
