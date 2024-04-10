import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
import zipfile

from matplotlib.patches import PathPatch
from scipy.interpolate import make_interp_spline
import math


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

main_df = pd.read_csv("Data_Staging.csv")

# display dataset
# print(df.head())
def summarization(df):
    # read the dataset using the compression zip
    # df = pd.read_csv('../DataStaging/Data_Staging.zip',compression='zip')
 


    # ======================  Curve Diagram: Annual Average AQI ======================
    # Group by year and month, then calculate the average AQI
    # Group by year and calculate the average AQI
    annual_avg_aqi = df.groupby('year')['AQI'].mean().reset_index()

    # Interpolation for a smoother line
    # X and Y values
    x = annual_avg_aqi['year']
    y = annual_avg_aqi['AQI']

    # Generate 300 points for a smoother curve
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)  # k is the degree of the spline
    y_smooth = spl(x_smooth)

    # Create a line plot for the average annual AQI with smoothing
    plt.figure(figsize=(12, 7))

    # Plotting the smooth curve
    plt.plot(x_smooth, y_smooth, color='blue')

    # Adding scatter plot for the actual data points for reference
    plt.scatter(x, y, color='red', alpha=0.6)

    # Adding titles and labels
    plt.title('Average Annual AQI Levels')
    plt.xlabel('Year')
    plt.ylabel('Average AQI')

    # Optionally, add grid lines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.show()

    # ======================  Histogram: AQI Categories ======================
    # Define the AQI categories 
    bins = [0, 50, 100, 150, 200, 300, 500]
    labels = ['Good', 'Moderate', 'Unhealthy(Sensitive Groups)', 'Unhealthy', 'Very Unhealthy', 'Hazardous']

    # Categorize the 'aqi' values
    tmpdf= pd.DataFrame()

    tmpdf['AQICategory'] = pd.cut(df['AQI'], bins=bins, labels=labels, include_lowest=True)
    aqi_category_counts = tmpdf['AQICategory'].value_counts(sort=False)

    # Create the histogram
    bars = plt.bar(aqi_category_counts.index, aqi_category_counts.values, color='blue', edgecolor='black')
    plt.title('Histogram of AQI Categories')
    plt.xlabel('AQI Category')
    plt.ylabel('Frequency')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=20)

    # Adding the total number on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, yval, ha='center', va='bottom')

    plt.show() # Show the histogram



    # ======================  Histogram: AQHI Categories  ======================
    # Define the AQHI categories 
    bins2 = [0,1,2,3,4]
    labels2 = ['Low', 'Moderate','High','Critical']

    # Categorize the 'aqhi' values
    tmpdf= pd.DataFrame()
    tmpdf['AQHICategory'] = pd.cut(df['AQHI'], bins=bins2, labels=labels2, include_lowest=True)
    aqhi_category_counts = tmpdf['AQHICategory'].value_counts(sort=False)

    # Create the histogram
    bars2 = plt.bar(aqhi_category_counts.index, aqhi_category_counts.values, color='yellow', edgecolor='black')
    plt.title('Histogram of AQHI Categories')
    plt.xlabel('AQHI Category')
    plt.ylabel('Frequency')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=20)

    # Adding the total number on top of each bar
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, yval, ha='center', va='bottom')

    plt.show() 

    # ======================  Histogram: Risk Level for Pollutant  ======================
    # Define the Risk Level categories 
    # Count the occurrences of each category
    category_counts = df['Category'].value_counts(sort=False)

    # Create the histogram
    bars = plt.bar(category_counts.index, category_counts.values, color='skyblue', edgecolor='black')
    plt.title('Histogram of Risk level for pollutant')
    plt.xlabel('Risk Category')
    plt.ylabel('Frequency')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)  # Adjust rotation for better readability of new labels

    # Adding the total number on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')  # Ensure yval is integer

    plt.tight_layout()  # Adjust layout to make sure everything fits without overlapping
    plt.show()

    # ======================  Histogram: Station Scale Type ======================
    # Count the frequency of each scale type
    scale_type_counts = df['Scale Type'].value_counts()

    # Create the histogram (bar chart)
    bars = plt.bar(scale_type_counts.index, scale_type_counts.values, color='skyblue', edgecolor='black')

    plt.title('Histogram of Station Scale Types')
    plt.xlabel('Scale Type')
    plt.ylabel('Frequency')

    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adding the total number on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')

    plt.tight_layout()  # Adjust the layout to make sure everything fits without overlapping
    plt.show()

    # ======================  Scatter plot: Weather Precipitation  ======================
    # for weather dimension
    # Group by year and then calculate the average precipitation
    annual_avg_prcp = df.groupby('year')['prcp'].mean().reset_index()
    # Now, plot the average annual precipitation
    # We will use the year for the x-axis and the average precipitation for the y-axis

    # Create a scatter plot
    plt.figure(figsize=(10, 6))  # Optional: defines the size of the figure

    # Scatter plot for the average annual precipitation
    plt.scatter(annual_avg_prcp['year'], annual_avg_prcp['prcp'], color='blue')

    # Adding titles and labels
    plt.title('Average Annual Precipitation')
    plt.xlabel('Year')
    plt.ylabel('Average Precipitation')

    # Show the scatter plot
    plt.show()


    # ======================  Scatter plot: AQI  ======================
    # Plot AQI level
    annual_avg_aqi = df.groupby('year')['AQI'].mean().reset_index()
    # Now, plot the average annual precipitation
    # We will use the year for the x-axis and the average precipitation for the y-axis

    # Create a scatter plot
    plt.figure(figsize=(10, 6))  # Optional: defines the size of the figure

    # Scatter plot for the average annual precipitation
    plt.scatter(annual_avg_aqi['year'], annual_avg_aqi['AQI'], color='red')

    # Adding titles and labels
    plt.title('Average Annual AQI ')
    plt.xlabel('Year')
    plt.ylabel('Average AQI')

    # Show the scatter plot
    plt.show()


    # ======================  Boxplot: Min and Max Temperature  ======================
    # Boxplot for max temperature & min temperature 
    # Find the record with the min temperature and the max temperature for each month of each year
    monthly_min = df.groupby(['year', 'month'])['tmin'].min().reset_index()
    monthly_max = df.groupby(['year', 'month'])['tmax'].max().reset_index()

    # Pivot the data to have years as rows and months as columns
    monthly_min_pivot = monthly_min.pivot(index='year', columns='month', values='tmin')
    monthly_max_pivot = monthly_max.pivot(index='year', columns='month', values='tmax')

    # Create a boxplot for each month with min and max temperatures
    plt.figure(figsize=(15, 10))

    # Generate positions for the month: [1, 2, ..., 12]
    positions = np.arange(1, 13)

    # Boxplot for min temperatures
    boxprops = dict(linestyle='-', linewidth=3, color='blue')
    medianprops = dict(linestyle='-', linewidth=2.5, color='skyblue')

    plt.boxplot([monthly_min_pivot[col].dropna() for col in monthly_min_pivot], 
                positions=positions - 0.15, widths=0.3, patch_artist=True,
                boxprops=boxprops, medianprops=medianprops, showfliers=False)

    # Boxplot for max temperatures
    boxprops.update(color='red')  # Update the properties for max temperature
    medianprops.update(color='darkred')

    plt.boxplot([monthly_max_pivot[col].dropna() for col in monthly_max_pivot], 
                positions=positions + 0.15, widths=0.3, patch_artist=True,
                boxprops=boxprops, medianprops=medianprops, showfliers=False)

    # Custom legend
    plt.legend([plt.Line2D([0], [0], color='blue', lw=4),
                plt.Line2D([0], [0], color='red', lw=4)],
            ['Min Temperature', 'Max Temperature'], loc='upper right')

    # Adding titles and labels
    plt.title('Monthly Min and Max Temperature Ranges (1980-2021)')
    plt.xlabel('Month')
    plt.ylabel('Temperature') 
    plt.xticks(positions, [f'{month}' for month in range(1, 13)])  # Month numbers as x-tick labels
    plt.xlim(0, 13)  
    plt.show()


    # ======================  Boxplot: CO Mean value  ======================
    # Boxplot for CO mean
    # Since the min and max value between other pollutant was really small difference
    # There we choose CO mean to do the visualization 
    # Calculate monthly averages for each year
    monthly_avg = df.groupby(['year', 'month'])['CO Mean'].mean().reset_index()

    # Find the year with max and min average CO value for each month
    monthly_max_avg = monthly_avg.loc[monthly_avg.groupby(['month'])['CO Mean'].idxmax()]
    monthly_min_avg = monthly_avg.loc[monthly_avg.groupby(['month'])['CO Mean'].idxmin()]

    # Combine max and min data for plotting
    combined_monthly_stats = pd.concat([monthly_max_avg, monthly_min_avg])

    # Pivot the combined data for easy plotting
    pivoted_data = combined_monthly_stats.pivot(index='month', columns='year', values='CO Mean')

    # Plotting
    plt.figure(figsize=(14, 10))
    positions = np.arange(1, 25, 2)  # Adjusted for 12 months
    boxplot = plt.boxplot(
        [pivoted_data.loc[month].dropna().values for month in range(1, 13)],
        positions=positions,
        patch_artist=True,
    )

    # Optional: Set colors for the boxplots
    # colors = ['skyblue' if (i % 2 == 0) else 'lightgreen' for i in range(12)]
    # for patch, color in zip(boxplot['boxes'], colors):
    #     patch.set_facecolor(color)

    # Adding labels and title for clarity
    plt.xlabel('Month')
    plt.ylabel('Average CO Mean Values (ppm)')
    plt.title('Distribution of Monthly Average CO Mean Values (1980-2021)')

    # Set x-axis tick labels to months
    plt.xticks(positions, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Optional: Add a grid for easier reading
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Display the plot
    plt.show()












# ================== Normalize Numeric Values ================== # 

def normalize_numeric(df, coloum_name):
    # column_no2 = 'NO2 Mean' 
    data = df[[coloum_name]]
        
        # Normalize
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data).round(2)

    # Add the normalized column to the df if it doesn't exist yet
    if ("Normalized "+coloum_name) not in df:
        df["Normalized "+coloum_name] = normalized_data

    print(df["Normalized "+coloum_name])
    return df

# ================== Hot Encdoing FactTable Values ================== # 

def Encoding(df, FeaturesDrop=[] ):
    categorical_features = ['season','day_of_week',"Status"]
    for i  in FeaturesDrop :
        if(i in categorical_features):
            categorical_features.remove(i)
        


    # # Initialize the OneHotEncoder
    encoder = OneHotEncoder(categories='auto',sparse=False)

    # # Fit and transform the data
    encoded_data = encoder.fit_transform(df[categorical_features])

    # # Create a DataFrame with the encoded features
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

    # # Concatenate the original DataFrame with the new one (excluding the original categorical columns)
    final_df = pd.concat([df.drop(categorical_features, axis=1), encoded_df], axis=1)

    # final_df.to_csv("encoding.csv", index=False)
    



    scale_type_order = ['Micro',"Middle", 'Neighbourhood', 'Urban', 'Regional']
    risk_level_order = ["Good","Moderate","Unhealthy for Sensitive Groups","Unhealthy","Very Unhealthy","Hazardous"]
    if("Timezone" not in FeaturesDrop):
        time_zone_level = ["America/Los_Angeles","America/Phoenix","America/Denver","America/Chicago","America/Indiana/Indianapolis","America/Detroit","America/New_York"]
        encoder_time_zone = OrdinalEncoder(categories=[time_zone_level])
        encoded_time_zone = encoder_time_zone.fit_transform(df[['Timezone']])
        final_df['Encoded Timezone'] = encoded_time_zone

    # Initialize the OrdinalEncoder with the categories in the defined order
    encoder = OrdinalEncoder(categories=[scale_type_order])
    encoder_risk_level = OrdinalEncoder(categories=[risk_level_order])

    # Fit and transform the "Scale Type" column
    # Note: reshape(-1, 1) is used because the encoder expects a 2D array
    encoded_scale_type = encoder.fit_transform(df[['Scale Type']])
    encoded_risk_level = encoder_risk_level.fit_transform(df[['Category']])


    # Add the encoded column back to the DataFrame, or create a new one as needed
    final_df['Encoded Scale Type'] = encoded_scale_type
    final_df['Encoded Category Type'] = encoded_risk_level
    return final_df


# def perform_enconding():
    # Specify the columns you want to encode

# final_df.to_csv("encoding.csv", index=False)

# print(final_df)


#================== Feature Selection ================== # 

def featureSelection(df):
    categorical_columns = ['City', 'State', 'Scale Type', 'Category','Timezone',"ID"]

    print(df.columns)
    X = df.drop(columns=categorical_columns)
    print(X.columns)

    # Initialize the VarianceThreshold object
    selector = VarianceThreshold(threshold=(.8 * (1 - .8)))

    # Fit and transform the data
    X_new = selector.fit_transform(X)

    # If you want to see which features were selected
    selected_features = X.columns[selector.get_support(indices=True)]

    # You can create a new DataFrame with the selected features if you want to keep the DataFrame structure
    X_new_df = pd.DataFrame(X_new, columns=selected_features)

    print("Selected features:", selected_features)
    df = main_df.drop(["Status","Timezone","day_of_week"],axis=1)
    
    #encode again
    result_df = Encoding(df, ["Status","Timezone","day_of_week"])
    return result_df
    # print(X_new_df.head())
    # print(X_new_df.columns)





def normalize(df):
    # # ================== Normalize Pollutant Values ================== # 
    print("Normalizing Pollutant")
    normalize_numeric( df,"NO2 Mean")
    normalize_numeric( df,"CO Mean")
    normalize_numeric( df,"SO2 Mean")
    normalize_numeric( df,"O3 Mean")
    normalize_numeric( df,"PM2.5 Mean")
    normalize_numeric( df,"PM10 Mean")
    print("Normalizing Pollutant FINISHED")


    # # ================== Normalize weather Values ================== # 

    print("Normalizing Weather")
    normalize_numeric( df,"tmax")
    normalize_numeric( df,"tmin")
    normalize_numeric( df,"prcp")
    print("Normalizing Weather FINISHED")


    # # ================== Normalize Geography Values ================== # 

    print("Normalizing Geography")
    normalize_numeric( df,"Population")
    normalize_numeric( df,"Population Density")
    print("Normalizing Geography FINISHED")


    # # ================== Normalize FactTable Values ================== # 

    print("Normalizing Fact Table")
    normalize_numeric( df,"AQI")
    result_df = normalize_numeric( df,"Visibility Range")
    print("Normalizing Fact Table FINISHED")
    return result_df



def find_outliers(df, feature_name):
    print(f"---[+] Identifying the outliers for {feature_name}...")
    start_time = time.time()
    X = df[[feature_name]]

    # Apply Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    df['Outlier'] = iso_forest.fit_predict(X)

    # Isolate outliers

    # Record the end time and calculate duration
    end_time = time.time()
    time_taken = end_time - start_time
    minutes = int(time_taken // 60)
    seconds = int(time_taken % 60)

    print(f"Time taken to calculate outliers for {feature_name}: {minutes} minutes and {seconds} seconds")
    plt.figure(figsize=(5, 6))
    # Visualize the outliers using a scatter plot
    # KDE plot for non-outliers
    sns.kdeplot(df[df['Outlier'] == 1]['AQI'], color="blue", fill=True, label='Normal', alpha=0.5)

# KDE plot for outliers
    sns.kdeplot(df[df['Outlier'] == -1]['AQI'], color="red", fill=True, label='Outliers', alpha=0.5)


    # Adding plot details
    plt.title('Density Plot of Normalized AQI')
    plt.xlabel('Normalized AQI')
    plt.ylabel('Density')
    plt.legend()

    plt.grid(True)  # Adding a grid for better readability
    plt.show()
    
# summarization(main_df)
result_df  = pd.DataFrame()
print("Encoding Data")
result_df = Encoding(main_df)
print("Feature Data")
result_df = featureSelection(result_df)
result_df = normalize(result_df)


result_df.to_csv("transformed_data.csv", index=False)

find_outliers(result_df,"AQI")
