import pandas as pd
import numpy as np
import datetime


#Question 9: Distance Matrix Calculations

def calculate_distance_matrix(csv_file):
    df = pd.read_csv(csv_file)
    
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    distance_matrix = pd.DataFrame(np.inf, index=ids, columns=ids)
    
    np.fill_diagonal(distance_matrix.values, 0)
    
    
    for _, row in df.iterrows():
        start, end, dist = row['id_start'], row['id_end'], row['distance']
        distance_matrix.at[start, end] = dist
        distance_matrix.at[end, start] = dist
    
    ids_list = distance_matrix.index
    for k in ids_list:
        for i in ids_list:
            for j in ids_list:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix

# Question 10: Unroll Distance Matrix

def unroll_distance_matrix(distance_matrix):
    unrolled = distance_matrix.reset_index().melt(id_vars='index', var_name='id_end', value_name='distance')
    
    unrolled = unrolled.rename(columns={'index': 'id_start'})
    

    unrolled = unrolled[unrolled['id_start'] != unrolled['id_end']]
    
    unrolled = unrolled[['id_end', 'id_start', 'distance']]
    unrolled.columns = [['id_start','id_end', 'distance']]
    
    return unrolled


# Question 11:  Finding IDs within Percentage Threshold

def find_ids_within_ten_percentage_threshold(df, reference_id):
    reference_rows = df[df['id_start'] == reference_id]
    
    reference_avg_distance = reference_rows['distance'].mean()

    lower_threshold = reference_avg_distance * 0.9  # 90% of the average
    upper_threshold = reference_avg_distance * 1.1  # 110% of the average

    avg_distances = df.groupby('id_start')['distance'].mean().reset_index()

    filtered_ids = avg_distances[
        (avg_distances['distance'] >= lower_threshold) & 
        (avg_distances['distance'] <= upper_threshold)
    ]['id_start']

    return sorted(filtered_ids.tolist())


# Question 12: Calculate Toll Rate


def calculate_toll_rate(df):
    # Define rate coefficients
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates and add new columns to the DataFrame
    df['moto'] = df['distance'] * rates['moto']
    df['car'] = df['distance'] * rates['car']
    df['rv'] = df['distance'] * rates['rv']
    df['bus'] = df['distance'] * rates['bus']
    df['truck'] = df['distance'] * rates['truck']

    return df



# Question 13:Calculate Time-Based Toll Rates




def calculate_time_based_toll_rates(df):
    # Define discount factors
    weekday_discount = {
        'morning': 0.8,  # 00:00 to 10:00
        'day': 1.2,      # 10:00 to 18:00
        'evening': 0.8   # 18:00 to 23:59
    }
    weekend_discount = 0.7  # For Saturday and Sunday

    # Days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create a list to hold the new rows
    new_rows = []

    # Generate entries for each unique (id_start, id_end) pair
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']  # Capture the distance from the input DataFrame

        # Iterate through each day of the week
        for day in days_of_week:
            # Calculate toll rates based on the day and apply appropriate discounts
            if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                # Weekdays
                toll_moto = row['moto'] * weekday_discount['morning']  # Morning discount
                toll_car = row['car'] * weekday_discount['morning']
                toll_rv = row['rv'] * weekday_discount['morning']
                toll_bus = row['bus'] * weekday_discount['morning']
                toll_truck = row['truck'] * weekday_discount['morning']
                
                # Create a row for morning rates
                new_rows.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': datetime.time(0, 0),
                    'end_day': day,
                    'end_time': datetime.time(10, 0),
                    'distance': distance,  # Add distance here
                    'moto': toll_moto,
                    'car': toll_car,
                    'rv': toll_rv,
                    'bus': toll_bus,
                    'truck': toll_truck
                })

                # Apply daytime rates
                toll_moto = row['moto'] * weekday_discount['day']  # Day discount
                toll_car = row['car'] * weekday_discount['day']
                toll_rv = row['rv'] * weekday_discount['day']
                toll_bus = row['bus'] * weekday_discount['day']
                toll_truck = row['truck'] * weekday_discount['day']

                # Create a row for daytime rates
                new_rows.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': datetime.time(10, 0),
                    'end_day': day,
                    'end_time': datetime.time(18, 0),
                    'distance': distance,  # Add distance here
                    'moto': toll_moto,
                    'car': toll_car,
                    'rv': toll_rv,
                    'bus': toll_bus,
                    'truck': toll_truck
                })

                # Apply evening rates
                toll_moto = row['moto'] * weekday_discount['evening']  # Evening discount
                toll_car = row['car'] * weekday_discount['evening']
                toll_rv = row['rv'] * weekday_discount['evening']
                toll_bus = row['bus'] * weekday_discount['evening']
                toll_truck = row['truck'] * weekday_discount['evening']

                # Create a row for evening rates
                new_rows.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': datetime.time(18, 0),
                    'end_day': day,
                    'end_time': datetime.time(23, 59, 59),
                    'distance': distance,  # Add distance here
                    'moto': toll_moto,
                    'car': toll_car,
                    'rv': toll_rv,
                    'bus': toll_bus,
                    'truck': toll_truck
                })
            else:
                # Weekends: Apply constant discount for all times
                discount = weekend_discount
                toll_moto = row['moto'] * discount
                toll_car = row['car'] * discount
                toll_rv = row['rv'] * discount
                toll_bus = row['bus'] * discount
                toll_truck = row['truck'] * discount

                # Create a row for the full 24-hour weekend period
                new_rows.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'start_day': day,
                    'start_time': datetime.time(0, 0),
                    'end_day': day,
                    'end_time': datetime.time(23, 59, 59),
                    'distance': distance,  # Add distance here
                    'moto': toll_moto,
                    'car': toll_car,
                    'rv': toll_rv,
                    'bus': toll_bus,
                    'truck': toll_truck
                })

    new_df = pd.DataFrame(new_rows)
    
    return new_df

# path = '../datasets/dataset-2.csv'
# matrix = calculate_distance_matrix(path)

# unrolled_matrix = unroll_distance_matrix(matrix)

# toll_df = calculate_toll_rate(unrolled_matrix)
# toll = calculate_time_based_toll_rates(toll_df)
# print(toll)