import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import dask.dataframe as dd
import time

from sklearn.linear_model import LinearRegression

#Import clean data
def import_cleandata():
    # Import data
    df = pd.read_csv("./Trips_Full Data.csv")
    ds = pd.read_csv("./Trips_by_Distance.csv")

    # Filter data where 'Level' is 'National'
    df = df[df['Level'] == 'National']
    ds = ds[ds['Level'] == 'National']

    # Convert 'Date' column to datetime type
    ds['Date'] = pd.to_datetime(ds['Date'], format='%m/%d/%Y')
    return df, ds 


#TASK 1 - a
def staying_home(data):
    start_time = time.time()
    # Group by year and week, then calculate the average population staying at home
    home_population_avg = data.groupby(pd.Grouper(key='Date', freq='W'))['Population Staying at Home'].mean()

    # Format the index to display year and week number
    week_year = [f"{idx.year} week {idx.weekofyear}" for idx in home_population_avg.index]

    # Plot the results
    plt.figure(figsize=(15, 6))  # Adjust the figure size as needed
    plt.plot(week_year, home_population_avg, marker='o', linestyle='-')
    plt.title('Average Population Staying at Home Over Time')
    plt.xlabel('Year-Week')
    plt.ylabel('Average Population')
    plt.xticks([week_year[i] if i % 6 == 0 else '' for i in range(len(week_year))], rotation=45)  # Show every other week
    plt.tight_layout()
    plt.show()
    
    # Calculate and print the time taken
    print("Time taken for staying_home(data):", time.time() - start_time, "seconds")

def distance_travel(data):
    # Select columns related to trip distances
    distance_columns = ['Trips <1 Mile', 'Trips 1-3 Miles', 'Trips 3-5 Miles', 'Trips 5-10 Miles',
                        'Trips 10-25 Miles', 'Trips 25-50 Miles', 'Trips 50-100 Miles',
                        'Trips 100-250 Miles', 'Trips 250-500 Miles', 'Trips 500+ Miles']

    # Calculate the mean number of trips for each distance category
    mean_trips_by_distance = data[distance_columns].mean()

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    mean_trips_by_distance.plot(kind='bar', color='skyblue', width = 0.9)
    plt.title('Average Number of People Travelling by Distance')
    plt.xlabel('Distance')
    plt.ylabel('Average Number of People Travelling')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


#TASK 1 - b
def compare_trip_categories(data):
    start_time = time.time()
    # Filter the dataset to identify dates where > 10,000,000 people conducted 10-25 trips
    trips_10_25 = data[data['Number of Trips 10-25'] > 10000000]

    # Filter the dataset to identify dates where > 10,000,000 people conducted 50-100 trips
    trips_50_100 = data[data['Number of Trips 50-100'] > 10000000]

    # Plot scatterplots
    plt.figure(figsize=(10, 6))

    # Scatter plot for 10-25 trips
    plt.subplot(2, 1, 1)
    plt.scatter(trips_10_25['Date'], trips_10_25['Number of Trips 10-25'], color='blue', label='10-25 Trips')
    plt.title('Dates with > 10,000,000 People Conducting 10-25 Trips')
    plt.xlabel('Date')
    plt.ylabel('Number of Trips')
    plt.xticks(rotation=45)
    plt.legend()

    # Scatter plot for 50-100 trips
    plt.subplot(2, 1, 2)
    plt.scatter(trips_50_100['Date'], trips_50_100['Number of Trips 50-100'], color='red', label='50-100 Trips')
    plt.title('Dates with > 10,000,000 People Conducting 50-100 Trips')
    plt.xlabel('Date')
    plt.ylabel('Number of Trips')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    # Calculate and print the time taken
    print("Time taken for compare_trip_categories(data):", time.time() - start_time, "seconds")


#TASK 1 - c
def staying_home_dask():
    #define number of processors
    n_processors=[10,20]
    n_processors_time={}
    for processor in n_processors:
        start_time=time.time()

        data = dd.read_csv('./Trips_by_Distance.csv',
                                   dtype={'County Name': 'object',
                                          'Number of Trips': 'float64',
                                          'Number of Trips 1-3': 'float64',
                                          'Number of Trips 10-25': 'float64',
                                          'Number of Trips 100-250': 'float64',
                                          'Number of Trips 25-50': 'float64',
                                          'Number of Trips 250-500': 'float64',
                                          'Number of Trips 3-5': 'float64',
                                          'Number of Trips 5-10': 'float64',
                                          'Number of Trips 50-100': 'float64',
                                          'Number of Trips <1': 'float64',
                                          'Number of Trips >=500': 'float64',
                                          'Population Not Staying at Home': 'float64',
                                          'Population Staying at Home': 'float64',
                                          'State Postal Code': 'object'})
        
        data['Date'] = dd.to_datetime(data['Date'], format='%m/%d/%Y')
        # Group by year and week, then calculate the average population staying at home
        home_population_avg = data['Date'].dt.strftime('%y %w')

        # Format the index to display year and week number
        #week_year = [f"{idx.year} week {idx.weekofyear}" for idx in home_population_avg.index]
        temp = data.compute(num_workers = processor)
        dask_time=time.time()-start_time
        n_processors_time[processor] = dask_time
    print(n_processors_time)

def compare_trips_dask():
    #define number of processors
    n_processors=[10,20]
    n_processors_time={}
    for processor in n_processors:
        start_time=time.time()

        data = dd.read_csv('./Trips_by_Distance.csv',
                                   dtype={'County Name': 'object',
                                          'Number of Trips': 'float64',
                                          'Number of Trips 1-3': 'float64',
                                          'Number of Trips 10-25': 'float64',
                                          'Number of Trips 100-250': 'float64',
                                          'Number of Trips 25-50': 'float64',
                                          'Number of Trips 250-500': 'float64',
                                          'Number of Trips 3-5': 'float64',
                                          'Number of Trips 5-10': 'float64',
                                          'Number of Trips 50-100': 'float64',
                                          'Number of Trips <1': 'float64',
                                          'Number of Trips >=500': 'float64',
                                          'Population Not Staying at Home': 'float64',
                                          'Population Staying at Home': 'float64',
                                          'State Postal Code': 'object'})
        
        data['Date'] = dd.to_datetime(data['Date'], format='%m/%d/%Y')
        # Filter the dataset to identify dates where > 10,000,000 people conducted 10-25 trips
        trips_10_25 = data[data['Number of Trips 10-25'] > 10000000]

        # Filter the dataset to identify dates where > 10,000,000 people conducted 50-100 trips
        trips_50_100 = data[data['Number of Trips 50-100'] > 10000000]
        temp = data.compute(num_workers = processor)
        dask_time=time.time()-start_time
        n_processors_time[processor] = dask_time
    print(n_processors_time)


#TASK 1 - d
def calculate_distance(row):
    distance_midpoints = {
        'Trips 1-25 Miles': 13,
        'Trips 25-100 Miles': 62.5,
        'Trips 100-250 Miles': 175,
        'Trips 250-500 Miles': 375,
        'Trips 500+ Miles': 750
    }
    total_distance = sum(row[category] * distance_midpoints[category] for category in distance_midpoints)
    return total_distance

    
    # Calculate the total distance by summing up the weighted distances
    total_distance = sum(row[category] * distance_midpoints[category] for category in distance_midpoints)
    return total_distance

def linear_regression(data, data_full):
    # For the first regression
    # Calculate the distance
    data_full['Distance'] = data_full.apply(calculate_distance, axis=1)
    x1 = data_full['People Not Staying at Home'].values.reshape(-1, 1)
    y1 = data_full['Distance'].values

    # For the second regression
    
    x2 = data['Population Not Staying at Home'].values.reshape(-1, 1)
    y2 = data['Number of Trips'].values

    # Splitting the data into training and testing sets for the first regression
    x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.2, random_state=42)

    # Initializing Linear Regression model for the first regression
    model1 = LinearRegression()

    # Training the first model
    model1.fit(x_train1, y_train1)

    # Evaluating the first model
    r_sq1 = model1.score(x_test1, y_test1)
    print("For Number of Trips:")
    print(f"Coefficient of determination (R^2): {r_sq1}")
    print(f"Intercept: {model1.intercept_}")
    print(f"Coefficients: {model1.coef_}")

    # Splitting the data into training and testing sets for the second regression
    x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.2, random_state=42)

    # Initializing Linear Regression model for the second regression
    model2 = LinearRegression()

    # Training the second model
    model2.fit(x_train2, y_train2)

    # Evaluating the second model
    r_sq2 = model2.score(x_test2, y_test2)
    print("For Distance:")
    print(f"Coefficient of determination (R^2): {r_sq2}")
    print(f"Intercept: {model2.intercept_}")
    print(f"Coefficients: {model2.coef_}")

    return (model1, r_sq1), (model2, r_sq2)


#TASK 1 - e
def travellers_by_distance(data):
    # Selecting relevant columns
    distance_columns = ['Trips 1-25 Miles', 'Trips 25-100 Miles', 'Trips 100-250 Miles',
                        'Trips 250-500 Miles', 'Trips 500+ Miles']
    distance_data = data[distance_columns]

    # Creating subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    # Plotting bar chart
    distance_data.sum().plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'],
                             ax=axes[0], width=0.9)
    axes[0].set_xlabel('Distance - Trips')
    axes[0].set_ylabel('Number of Travellers')
    axes[0].set_title('Number of Travellers by Distance-Trips')
    axes[0].tick_params(axis='x', rotation=45)

    # Plotting line plot
    distance_data.sum().plot(kind='line', marker='o', color='orange', ax=axes[1])
    axes[1].set_xlabel('Distance - Trips')
    axes[1].set_ylabel('Number of Travellers')
    axes[1].set_title('Number of Travellers by Distance-Trips')
    axes[1].grid(True)
    axes[1].tick_params(axis='x', rotation=45)

    # Adjust layout
    plt.tight_layout()
    plt.show()


#Main function
if __name__ == "__main__":
    df, ds = import_cleandata()
    #staying_home(ds)
    #distance_travel(df)
    
    #compare_trip_categories(ds)
    
    #staying_home_dask()
    #compare_trips_dask()
    
    linear_regression(ds, df)
    
    #travellers_by_distance(df)