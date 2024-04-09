CREATE TABLE Geography (
    geography_id SERIAL PRIMARY KEY,
    State VARCHAR NOT NULL,
    City VARCHAR NOT NULL,
    Timezone VARCHAR NOT NULL,
    Population INTEGER NOT NULL,
    Population_Density INTEGER NOT NULL,
    Area_Code INTEGER NOT NULL,
    CONSTRAINT unique_geo UNIQUE (State, City,Timezone,Area_Code,Population,Population_Density)
);

CREATE TABLE Date (
    date_id SERIAL PRIMARY KEY,
    day INTEGER NOT NULL,
    month INTEGER NOT NULL,
    year INTEGER NOT NULL,
    day_of_the_week VARCHAR NOT NULL,
    season VARCHAR NOT NULL,
    CONSTRAINT unique_date UNIQUE (day, month, year, day_of_the_week,season)
);

CREATE TABLE Pollutant (
    pollutant_id INTEGER PRIMARY KEY,
    pm25 FLOAT NOT NULL,
    pm10 INTEGER NOT NULL,
    O3_Mean FLOAT NOT NULL,
    CO_Mean FLOAT NOT NULL,
    SO2_Mean FLOAT NOT NULL,
    NO2_Mean FLOAT NOT NULL,
    Risk_Level VARCHAR NOT NULL
);

CREATE TABLE Station (
    station_id SERIAL PRIMARY KEY,
	ID VARCHAR NOT NULL,
    Scale_Type VARCHAR NOT NULL,
    Status VARCHAR NOT NULL,
    CONSTRAINT unique_station UNIQUE (ID, Scale_Type, Status)
);

CREATE TABLE Weather (
    weather_id SERIAL PRIMARY KEY,
    tmax INTEGER NOT NULL,
    tmin INTEGER NOT NULL,
    prcp FLOAT NOT NULL
);

-- Fact Table
CREATE TABLE AirQuality (
    PRIMARY KEY (geography_id, date_id, pollutant_id, station_id, weather_id),
    geography_id BIGINT REFERENCES Geography(geography_id),
    date_id BIGINT REFERENCES Date(date_id),
    pollutant_id BIGINT REFERENCES Pollutant(pollutant_id),
    station_id BIGINT REFERENCES Station(station_id),
    weather_id BIGINT REFERENCES Weather(weather_id),
    aqi INTEGER NOT NULL,
    aqhi INTEGER NOT NULL,
    visibility_range FLOAT NOT NULL
    
);