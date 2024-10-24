# Question 1:

# Problem Statement: Write a function that takes a list and an integer n,and returns the list with every group of n elements reversed. 
#                     If there are fewer than n elements left at the end, reverse all of them.

def reverse_list(lst,n):
    result = []
    i =  0
    while i < len(lst):
        if i + n <= len(lst):
            j = i + n - 1
            counter = 0
            while counter < n:
                result.append(lst[j])
                j -= 1
                counter += 1
        else:
            j = len(lst) - 1
            while j >= i:
                result.append(lst[j])
                j -= 1
        i += n
        
    return result


# Question 2:

# Problem Statement: Write a function that takes a list of strings and groups them by their length. 
#                    The result should be a dictionary

def list_to_dict(lst):
    length =set()
    for i in lst:
        length.add(len(i))
    length = sorted(length)
    result = {key: [] for key in length}
    for i in lst:
        for key in result.keys():
            if len(i) == key:
                result[key].append(i)
    return result


# Question 3:

# Problem Statement: You are given a nested dictionary that contains various details (including lists and sub-dictionaries). Your task is to write a Python function that flattens the dictionary such that:
#                     Nested keys are concatenated into a single key with levels separated by a dot (.).
#                     List elements should be referenced by their index, enclosed in square brackets (e.g., sections[0]).
#                     For example, if a key points to a list, the index of the list element should be appended to the key string, followed by a dot to handle further nested dictionaries.

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Construct the new key
        if isinstance(v, dict): 
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list): 
            for i, item in enumerate(v):
                list_key = f"{new_key}[{i}]"
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, list_key, sep=sep).items())
                else:
                    items.append((list_key, item))
        else:
            items.append((new_key, v))
    return dict(items)


# Question 4 : 

# Problem Statement:You are given a list of integers that may contain duplicates. 
#                   Your task is to generate all unique permutations of the list. 
#                   The output should not contain any duplicate permutations.

def permute_unique(nums):
    result = []
    nums.sort()
    result.append(nums[:])

    while True:
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        
        if i == -1:
            break
        
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        
        nums[i], nums[j] = nums[j], nums[i]
        nums[i + 1:] = reversed(nums[i + 1:])
        result.append(nums[:])

    return result


# Question 5:

# Problem Statement:You are given a string that contains dates in various formats
#                     (such as "dd-mm-yyyy", "mm/dd/yyyy", "yyyy.mm.dd", etc.). 
#                    Your task is to identify and return all the valid dates present in the string.


import re

def find_all_dates(txt):
    pattern = r'[0-9]{2}-[0-9]{2}-[0-9]{4}|[0-9]{2}/[0-9]{2}/[0-9]{4}|[0-9]{4}\.[0-9]{2}\.[0-9]{2}'
    result = re.findall(pattern, txt)
    return result


# Question 6: Decode Polyline, Convert to DataFrame with Distances

import polyline
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Haversine formula to calculate distance between two lat-lon pairs
def haversine(lat1, lon1, lat2, lon2):
    # Earth radius in meters
    R = 6371000  
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c  # Distance in meters

# Function to decode polyline and calculate distances
def decode_polyline_to_dataframe(polyline_str):
    # Step 1: Decode the polyline string
    coordinates = polyline.decode(polyline_str)
    
    # Step 2: Create a DataFrame with latitude and longitude
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Step 3: Calculate distance between consecutive points
    distances = [0]  # First point has no previous point, so distance is 0
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    # Step 4: Add the distance column to the DataFrame
    df['distance'] = distances
    
    return df

# Example usage
# polyline_str = "}_p~F~ps|U_ulLnnqC_mqNvxq`@"  # Replace with your polyline string
# df = decode_polyline_to_dataframe(polyline_str)
# print(df)


# Question 7: Matrix Rotation and Transformation

import numpy as np

def transposed_matrix(matrix):
    matrix_t = []
    for i in range(len(matrix)):
        temp_lst = []
        j = len(matrix) - 1
        while j>=0:
            temp_lst.append(matrix[j][i])
            j-=1
        matrix_t.append(temp_lst)

    mat = np.array(matrix_t)
    n, m = mat.shape
    result = np.zeros((n, m), dtype=int)

    row_sums = np.sum(mat, axis=1)
    col_sums = np.sum(mat, axis=0)
    for i in range(n):
            for j in range(m):
                result[i][j] = row_sums[i] + col_sums[j] - (2 * mat[i][j])

    return result

# Question 8: Time Check

def check_timestamp_completeness(df):
    # Create a function to convert days into actual dates within a week
    def convert_days_to_dates(start_day, week_start='2023-01-02'):
        # Create a mapping from day names to dates
        day_to_date = {
            'Monday': pd.to_datetime(week_start),  # Start of the week
            'Tuesday': pd.to_datetime(week_start) + pd.DateOffset(days=1),
            'Wednesday': pd.to_datetime(week_start) + pd.DateOffset(days=2),
            'Thursday': pd.to_datetime(week_start) + pd.DateOffset(days=3),
            'Friday': pd.to_datetime(week_start) + pd.DateOffset(days=4),
            'Saturday': pd.to_datetime(week_start) + pd.DateOffset(days=5),
            'Sunday': pd.to_datetime(week_start) + pd.DateOffset(days=6),
        }
        return day_to_date[start_day]

    # Convert the day strings to actual dates and concatenate with times
    df['start_timestamp'] = df.apply(lambda x: convert_days_to_dates(x['startDay']) + pd.to_timedelta(x['startTime']), axis=1)
    df['end_timestamp'] = df.apply(lambda x: convert_days_to_dates(x['endDay']) + pd.to_timedelta(x['endTime']), axis=1)

    # Function to check if a pair covers full 7 days with 24-hour periods
    def is_complete(group):
        # Get the set of days covered (e.g., "Monday", "Tuesday", etc.)
        days_covered = set(group['startDay'])
        
        # Ensure all 7 days are covered
        full_week = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        if days_covered != full_week:
            return False
        
        # Check 24-hour coverage for each day
        for day in full_week:
            day_records = group[group['startDay'] == day]
            if not check_24_hour_coverage(day_records):
                return False
        
        return True

    # Function to check if a given day's timestamps cover the full 24 hours
    def check_24_hour_coverage(group):
        # We will iterate through the rows and ensure the entire 24-hour period is covered
        # Starting from 00:00:00 to 23:59:59
        time_intervals = [False] * (24 * 60)  # Each minute of the day
        
        for _, row in group.iterrows():
            start_time = row['start_timestamp']
            end_time = row['end_timestamp']
            
            # Mark minutes in the time_intervals array as True if they are covered by the timestamps
            start_minute = start_time.hour * 60 + start_time.minute
            end_minute = end_time.hour * 60 + end_time.minute

            for minute in range(start_minute, end_minute + 1):
                if minute < len(time_intervals):
                    time_intervals[minute] = True
        
        # Check if all minutes of the day are covered
        return all(time_intervals)

    # Group by (id, id_2) and apply the completeness check
    completeness = df.groupby(['id', 'id_2']).apply(is_complete)

    # Return the result as a boolean series with multi-index
    return completeness

# import pandas as pd
# df = pd.read_csv('../datasets/dataset-1.csv')
# print(check_timestamp_completeness(df))