import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from scipy.interpolate import make_interp_spline
import math


# display dataset
# print(df.head())

# read the dataset using the compression zip
# df = pd.read_csv('../DataStaging/Data_Staging.zip',compression='zip')
df = pd.read_csv('../DataStaging/Data_Staging.csv')


# ======================  Curve Diagram: Annual avaerage AQI ======================
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

# ======================  Histogram: AQI categories ======================
# Define the AQI categories 
bins = [0, 50, 100, 150, 200, 300, 500]
labels = ['Good', 'Moderate', 'Unhealthy(Sensitive Groups)', 'Unhealthy', 'Very Unhealthy', 'Hazardous']

# Categorize the 'aqi' values
df['AQICategory'] = pd.cut(df['AQI'], bins=bins, labels=labels, include_lowest=True)
aqi_category_counts = df['AQICategory'].value_counts(sort=False)

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



# ======================  Histogram: AQHI categories  ======================
# Define the AQHI categories 
bins2 = [0,1,2,3,4]
labels2 = ['Low', 'Moderate','High','Critical']

# Categorize the 'aqhi' values
df['AQHICategory'] = pd.cut(df['AQHI'], bins=bins2, labels=labels2, include_lowest=True)
aqhi_category_counts = df['AQHICategory'].value_counts(sort=False)

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

# ======================  Scatter plot: weather precipitation  ======================
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


# ======================  Boxplot: Min and Max temperature  ======================
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