import json
import re

def main():
    signalsList = []
    with open('signals.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            _, signals = line.split(": ")
            str_list = signals[1:-1].split(',')
            lst = [x.strip() for x in str_list]
            signalsList.append(lst)
        f.close()
        
    signalDict = {}
    for signalList in signalsList:
        for signal in signalList:
            if signal not in signalDict:
                signalDict[signal] = 1
            else:
                signalDict[signal] += 1
            
    with open('signals.csv', 'w', encoding='utf-8') as f:
        for signal in signalDict:
            clean_signal = re.sub(r"[\n\t]+", " ", signal.replace("\\n", " ").replace("\\t", " "))
            f.write(f"{clean_signal},{signalDict[signal]}\n")
        
if __name__ == "__main__":
    main()
