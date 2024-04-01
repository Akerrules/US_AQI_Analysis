import pandas as pd
import psycopg2
from psycopg2 import extras


conn = psycopg2.connect(
    dbname="datascience",
    user="postgres",
    password="",
    host="localhost",
    port="5432"
)
cur = conn.cursor()
df = pd.read_csv('Data_Staging.csv')
print("getting started uploaded")

for index, row in df.iterrows():
    cur.execute("""
        INSERT INTO Geography (geography_id, City, State, Population, Population_Density, Timezone, Area_Code)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (row['Surrogate Keys'], row['City'], row['State'], row['Population'], row['Population Density'], row['Timezone'], row['Area Code']))
print("Geo uploaded")

# Insert data into the DateInfo table
for index, row in df.iterrows():
    cur.execute("""
        INSERT INTO DateInfo (date_id, day, month, year, season, day_of_the_week)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (row['Surrogate Keys'], row['day'], row['month'], row['year'], row['season'], row['day_of_week']))
print("Date uploaded")

# Insert data into the Weather table
for index, row in df.iterrows():
    cur.execute("""
        INSERT INTO Weather (weather_id, tmax, tmin, prcp)
        VALUES (%s, %s, %s, %s)
    """, (row['Surrogate Keys'], row['tmax'], row['tmin'], row['prcp']))
print("weather uploaded")

# Insert data into the Station table
for index, row in df.iterrows():
    cur.execute("""
        INSERT INTO Station (station_id, ID, Scale_Type, Status)
        VALUES (%s, %s, %s, %s)
    """, (row['Surrogate Keys'], row['ID'], row['Scale Type'], row['Status']))
print("Station uploaded")

# Insert data into the Pollutant table
for index, row in df.iterrows():
    cur.execute("""
        INSERT INTO Pollutant (pollutant_id, NO2_Mean, CO_Mean, SO2_Mean, O3_Mean, pm25, pm10, Risk_Level)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (row['Surrogate Keys'], row['NO2 Mean'], row['CO Mean'], row['SO2 Mean'], row['O3 Mean'], row['PM2.5 Mean'], row['PM10 Mean'], row['Category']))
print(" Pollutant uploaded")

# Insert data into the AQI table
for index, row in df.iterrows():
    cur.execute("""
        INSERT INTO AQI (aqi_id, geography_id, date_id, weather_id, station_id, pollutant_id, aqi, aqhi, visibility_range)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (row['Surrogate Keys'], row['Surrogate Keys'], row['Surrogate Keys'], row['Surrogate Keys'], row['Surrogate Keys'], row['Surrogate Keys'], row['AQI'], row['AQHI'], row['Visibility Range']))

print("done")
conn.commit()
cur.close()
conn.close()
