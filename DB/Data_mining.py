import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest


sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

main_df = pd.read_csv("Data_Staging.csv")



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
    plt.figure(figsize=(10, 6))
    # Visualize the outliers using a scatter plot
    # KDE plot for non-outliers
    sns.kdeplot(df[df['Outlier'] == 1]['Normalized AQI'], color="blue", fill=True, label='Normal', alpha=0.5)

# KDE plot for outliers
    sns.kdeplot(df[df['Outlier'] == -1]['Normalized AQI'], color="red", fill=True, label='Outliers', alpha=0.5)


    # Adding plot details
    plt.title('Density Plot of Normalized AQI')
    plt.xlabel('Normalized AQI')
    plt.ylabel('Density')
    plt.legend()

    plt.grid(True)  # Adding a grid for better readability
    plt.show()
    

result_df  = pd.DataFrame()
print("Encoding Data")
result_df = Encoding(main_df)
print("Feature Data")
result_df = featureSelection(result_df)
result_df = normalize(result_df)


result_df.to_csv("transformed_data.csv", index=False)

find_outliers(result_df,"AQI")