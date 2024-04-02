import pandas as pd
import psycopg2
from psycopg2 import extras
import json

conn = psycopg2.connect(
    dbname="datascience",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)
cur = conn.cursor()
df = pd.read_csv('Data_Staging.csv')





print("getting started uploaded")

# geo_tuples = list(df[[ 'City', 'State', 'Population', 'Population Density', 'Timezone', 'Area Code']].itertuples(index=False, name=None))
date_tuples = list(df[[ 'day', 'month', 'year', 'season', 'day_of_week']].itertuples(index=False, name=None))
# weather_tuples = list(df[['Surrogate Keys', 'tmax', 'tmin', 'prcp']].itertuples(index=False, name=None))
# station_tuples = list(df[['ID', 'Scale Type', 'Status']].itertuples(index=False, name=None))
# pollutant_tuples = list(df[['Surrogate Keys', 'NO2 Mean', 'CO Mean', 'SO2 Mean', 'O3 Mean', 'PM2.5 Mean', 'PM10 Mean', 'Category']].itertuples(index=False, name=None))
# # fact_tuple = list(df[['Surrogate Keys', 'Surrogate Keys', 'Surrogate Keys', 'Surrogate Keys', 'Surrogate Keys', 'Surrogate Keys', 'AQI', 'AQHI', 'Visibility Range']].itertuples(index=False, name=None))
# print("tuppling done")



def upload_geo():
    extras.execute_batch(cur,"""
        INSERT INTO Geography (City, State, Population, Population_Density, Timezone, Area_Code)
        VALUES ( %s, %s, %s, %s, %s, %s)
        ON CONFLICT (City, State, Population, Population_Density, Timezone,Area_Code) DO NOTHING;
    """, geo_tuples)

    print("Geo uploaded")
    conn.commit()


def upload_date():
    # Insert data into the Date table
    extras.execute_batch(cur,"""
            INSERT INTO Date (day, month, year, season, day_of_the_week)
            VALUES (%s, %s, %s, %s, %s)  
            ON CONFLICT (day,month,year,season, day_of_the_week ) DO NOTHING
    """
    , date_tuples)

    conn.commit()
    print("Date uploaded")


# # Insert data into the Weather table
def upload_weather():
    extras.execute_batch(cur,""" INSERT INTO Weather (weather_id, tmax, tmin, prcp)  VALUES (%s, %s, %s, %s)
        """, weather_tuples)

    conn.commit()
    print("weather uploaded")


# # Insert data into the Station table
def upload_station():
    extras.execute_batch(cur,"""
            INSERT INTO Station ( ID, Scale_Type, Status)
            VALUES ( %s, %s, %s)
            ON CONFLICT (ID, Scale_Type, Status) DO NOTHING 
        """, station_tuples)
    
    conn.commit()
    print("Station uploaded")

# # Insert data into the Pollutant table
def upload_pollutant():
    extras.execute_batch(cur,"""
            INSERT INTO Pollutant (pollutant_id, NO2_Mean, CO_Mean, SO2_Mean, O3_Mean, pm25, pm10, Risk_Level)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, pollutant_tuples)

    conn.commit()
    print(" Pollutant uploaded")

# # Insert data into the AQI table

def upload_fact(fact_tuple):
    extras.execute_batch(cur,"""
        INSERT INTO AirQuality (date_id,geography_id,station_id , pollutant_id, weather_id,visibility_range,aqi,aqhi)
        VALUES (%s, %s, %s, %s, %s,%s,%s,%s)
    """, fact_tuple)


def getkeys():
    cache_keys = {
    "weather":{},
    "date":{},
    'pollutant':{},
    'station':{},
    'geo':{}

    }
    cur.execute("SELECT pollutant_id FROM Pollutant;")
    for key in cur.fetchall():  # returns a tuplein the format (id,)
        cache_keys["pollutant"][key[0]] = key[0]

    
    cur.execute("SELECT weather_id FROM Weather")
    for key in cur.fetchall():
        cache_keys['weather'][key[0]] = key[0]
    
    
    cur.execute("SELECT station_id, ID,Scale_Type, Status FROM Station")
    for key, ID , Scale_Type, Status in cur.fetchall():
        cache_keys['station'][(ID, Scale_Type, Status)] = key
    

    cur.execute("SELECT geography_id, City, State, Population, Population_Density, Timezone, Area_Code FROM Geography ")
    for key, City, State, Population, Population_Density, Timezone, Area_Code in cur.fetchall():
        cache_keys['geo'][(City, State, Population, Population_Density, Timezone, Area_Code)] = key
    

    cur.execute("SELECT date_id, day, month, year, season, day_of_the_week FROM Date")
    for key, day, month, year, season, day_of_the_week in cur.fetchall():
        cache_keys['date'][(day, month, year, season, day_of_the_week)] = key
    
    # print(cache_keys['date'])
    # print(cache_keys['date'][(1, '1', 1980, 'Winter', 'Tuesday')])
    return cache_keys

def prep_Fact_Data(cache_keys):
    data_to_fact=[]
    for index, row in df.iterrows():
        pollutant_key = cache_keys['pollutant'][row['Surrogate Keys']]
        weather_key = cache_keys['weather'][row['Surrogate Keys']]
        station_key = cache_keys['station'].get((row['ID'], row['Scale Type'], row['Status']))
        geo_key = cache_keys['geo'].get((row['City'],row['State'],row['Population'],row['Population Density'],row['Timezone'],row['Area Code']))
        date_key = cache_keys['date'].get((row['day'], row['month'], row['year'], row['season'], row['day_of_week']))

        # print([ date_key],cache_keys['date'][(row['day'], row['month'], row['year'], row['season'], row['day_of_week'])])
        # print([pollutant_key,weather_key, station_key, geo_key, date_key])
        if all([pollutant_key,weather_key, station_key, geo_key, date_key]):
            data_to_fact.append((date_key,geo_key,station_key,pollutant_key,weather_key, row['Visibility Range'],row['AQI'],row['AQHI']))

    return data_to_fact

# upload_geo()
# upload_pollutant()
# upload_station()
# upload_date()
# upload_weather()

cache_keys = getkeys()
fact_tuples = prep_Fact_Data(cache_keys)
upload_fact(fact_tuples)

conn.commit()
print("done")

cur.close()
conn.close()
