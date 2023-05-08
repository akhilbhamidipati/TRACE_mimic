import os

def get_intervals():
    """Retrieves intervals for each file of all patients

    Returns:
        list<int>: List of intervals, each interval is for each file of each patient 
    """
    intervals = []
    signal_types = set()
    interval_signal_map = {}

    for folder_name in os.listdir(os.getcwd()):
        if folder_name.isdigit():
            print(f"Getting intervals for patient name: {folder_name}")
            # loop through every file
            inter, sign = get_interval_for_each_record(folder_name)
            intervals.append(inter)
            signal_types = signal_types.union(sign)
            interval_signal_map[folder_name]  = sign
    return intervals, signal_types, interval_signal_map

def get_interval_for_each_record(folder_name):
    file_names = os.listdir(os.path.join(os.getcwd(), folder_name))
    for file_name in file_names:
        if file_name.endswith((".txt")):
            file_path = os.path.join(os.getcwd(), folder_name, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                signal_types = set()
                line_count = 0
                intervals = 0
                for line in lines:
                    if line_count == 0:
                        line_count += 1
                        continue
                    prefix = line.split(" ")[0]
                    #print(prefix)
                    signal_types.add(prefix)
                    if prefix in signal_types:
                        number_of_signals = len(signal_types)
                        if (line_count+1) % number_of_signals == 0:
                            intervals += 1
                    line_count += 1
    return intervals, signal_types

def print_intervals(intervals):
    interval_set = set()
    for interval in intervals:
        interval_set.add(interval)
    if len(interval_set) > 1:
        print("These are the different intervals found:")
        for interval in interval_set:
            print(f"{interval} ", end=" ")
        print(sum(interval_set) / len(interval_set))
    elif len(interval_set) == 1:
        interval_constant = interval_set.pop()
        print(f"There is only one interval: {interval_constant}")
    else:
        print("Something went wrong")
    
        
def print_signals(signal_types):
    for signal in signal_types:
        print(signal.strip(), end=" ")

def main():
    intervals, signal_types, interval_signal_map = get_intervals()
    print_intervals(intervals)
    print_signals(signal_types)
    for patient_name, signals in interval_signal_map.items():
        print(f"{patient_name} has these signals: {[signal for signal in signals]}")

if __name__ == "__main__":
    main()
