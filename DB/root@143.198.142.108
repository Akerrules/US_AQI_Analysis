CREATE TABLE Geography (
    geography_id INTEGER PRIMARY KEY,
    State VARCHAR NOT NULL,
    City VARCHAR NOT NULL,
    Timezone VARCHAR NOT NULL,
    Population INTEGER NOT NULL,
    Population_Density INTEGER NOT NULL,
    Area_Code INTEGER NOT NULL
);

CREATE TABLE Date (
    date_id INTEGER PRIMARY KEY,
    day INTEGER NOT NULL,
    month VARCHAR NOT NULL,
    year INTEGER NOT NULL,
    day_of_the_week VARCHAR NOT NULL,
    season VARCHAR NOT NULL
);

CREATE TABLE Pollutant (
    pollutant_id INTEGER PRIMARY KEY,
    pm25 FLOAT NOT NULL,
    pm10 INTEGER NOT NULL,
    O3_Mean FLOAT NOT NULL,
    CO_Mean FLOAT NOT NULL,
    SO2_Mean INTEGER NOT NULL,
    NO2_Mean INTEGER NOT NULL,
    Risk_Level VARCHAR NOT NULL
);

CREATE TABLE Station (
    station_id INTEGER PRIMARY KEY,
	ID VARCHAR NOT NULL,
    Scale_Type VARCHAR NOT NULL,
    Status VARCHAR NOT NULL
);

CREATE TABLE Weather (
    weather_id INTEGER PRIMARY KEY,
    tmax INTEGER NOT NULL,
    tmin INTEGER NOT NULL,
    prcp FLOAT NOT NULL
);

-- Fact Table
CREATE TABLE AirQuality (
    aqi_id SERIAL PRIMARY KEY,
    geography_id INTEGER REFERENCES Geography(geography_id),
    date_id INTEGER REFERENCES Date(date_id),
    pollutant_id INTEGER REFERENCES Pollutant(pollutant_id),
    station_id INTEGER REFERENCES Station(station_id),
    weather_id INTEGER REFERENCES Weather(weather_id),
    aqi INTEGER NOT NULL,
    aqhi INTEGER NOT NULL,
    visibility_range INTEGER NOT NULL
);