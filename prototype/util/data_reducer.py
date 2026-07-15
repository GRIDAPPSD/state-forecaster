#!/usr/bin/python3

import json

file_path = 'results_data_forecasting_13.json'

# Open the file and process it line by line
with open(file_path, 'r', encoding='utf-8') as file:
    #redSec = 300
    redSec = 900
    allCount = 0
    redCount = 0
    for line in file:
        # Strip whitespace and skip empty lines
        clean_line = line.strip()
        if clean_line:
            # Parse the line into a Python dictionary
            json_object = json.loads(clean_line)

            timestamp = json_object['timestamp']
            allCount += 1
            if timestamp % redSec == 0:
              redCount += 1
              print(clean_line)

#print("All count: " + str(allCount))
#print("Reduced count: " + str(redCount))
