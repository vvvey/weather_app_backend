from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import math
import logging
import os

import psycopg2
from keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
load_dotenv()

# API Keys and Endpoint
ENDPOINT = "https://rt.ambientweather.net/v1/devices?"
API_KEY = '80d0cbf141844d2e8afdd8dd4caf9d6979393c605d85477386ceff64899bb067'
APP_KEY = 'a784dc6788c74457b7d77bbe095f1b9f9488b233772d4d92bab211b9ea3bac85'
# API_KEY = os.getenv('API_KEY')
# APP_KEY = os.getenv('APP_KEY')

DB_HOST = "hyperlocal-db.c41iwuymw07w.us-east-1.rds.amazonaws.com" 
DB_NAME = "weather_db"
DB_USER = "postgres"
DB_PASSWORD = "Team1isfire!"  
DB_PORT = "5432" 

logging.basicConfig(level=logging.DEBUG)

ice_model = load_model('./model_lstm.keras')
ice_scaler = joblib.load('./scaler_lstm.pkl')

def f_to_c(tempf):
    return (tempf - 32) * (5 / 9)

def Hossain_2014(H, p, Ce, Te, T, B, BPRT):
    try:
        denominator = (Te - Ce * T) * (T - Te) * BPRT * B
        if denominator == 0:
            return None
        elif T > 1:
            return 0 
        else:
            R =  math.log(0.2) * Te * Ce * T * p * (H / denominator)
            return max(0, R)
    except (ZeroDivisionError, ValueError, TypeError):
        return None

def get_db_connection():
    """
    Function to create and return a database connection
    """
    try:
        # Establish the connection
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        raise

def get_weather_records(last_n=10):
    """
    Function to get the last n weather data records from the weather_t table
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        select_query = f"""
        SELECT tempf, humidity, windspeedmph, windgustmph, winddir, timestamp, (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) / 60.0 AS tdiff FROM weather_t ORDER BY timestamp DESC LIMIT {last_n};
        """
        cur.execute(select_query)
        data = cur.fetchall()

        cur.close()
        return data
    except Exception as e:
        print(f"Error retrieving data: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")

def predict(data):
    """
    Function to predict if ice will form based on the given weather data
    Return the probability of ice formation
    """

    # global ice_model, ice_scaler
    global ice_model, ice_scaler

    # Some data transformations
    transformed_data = []
    for row in data:
        tmpf = float(row[0])
        humidity = float(row[1])
        windspeedmph = float(row[2])
        windgustmph = float(row[3])
        winddir = int(row[4])
        timestamp = int(row[5])
        tdiff = float(row[6])


        isday = 1 if (timestamp % 86400) >= 21600 and (timestamp % 86400) <= 64800 else 0
        windspeedknots = float(windspeedmph) * 0.868976
        windgustknots = float(windgustmph) * 0.868976

        new_row = [tmpf, humidity, windspeedknots, windgustknots, tdiff, isday]
        transformed_data.append(new_row)

     # reverse the order (newest last)
    transformed_data = transformed_data[::-1]
    
    features = ['tmpf', 'rh', 'sknt', 'gust', 'tdiff', 'isday']
    sample = np.array(transformed_data)
    sequence_length = len(transformed_data)

    # Normalize
    X = ice_scaler.transform(sample)

    # Predict
    y_pred = ice_model.predict(X.reshape(1, sequence_length, len(features)))
    return float(y_pred[0][0])

@app.route('/api/weather')
def get_weather():
    requests 
    try:
        response = requests.get(f"{ENDPOINT}applicationKey={APP_KEY}&apiKey={API_KEY}")
        response.raise_for_status()
        data = response.json()
        weather_data = [
            {
                "lastData": {
                    "tempf": data[0]["lastData"]["tempf"]
                }
            }
        ]
        # Store the temperature data in a global variable
        global last_tempf
        last_tempf = data[0]["lastData"]["tempf"]
        return jsonify(weather_data)
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching weather data: {e}")

@app.route('/api/weather/hossain')
def get_hossain():
    temp = f_to_c(last_tempf)
    
    app.logger.info(f"Temperature C: {temp}")

    result, green_result, jb_result, slicer_result, blue_result = 0,0,0,0,0
    try:
        # Use the stored temperature data
        sample_data = {
            'snow_height': 0.25,
            'snow_density': 300,
            'salt_concentration': 0.23,
            'salt_temp': -21,
            'pave_temp': temp,
            'melt_speed': 0.49,
            'BPRT': 2.04
        }
        result = Hossain_2014(
            sample_data['snow_height'],
            sample_data['snow_density'],
            sample_data['salt_concentration'],
            sample_data['salt_temp'],
            sample_data['pave_temp'],
            sample_data['melt_speed'],
            sample_data['BPRT']
        )

        app.logger.info(f"Result: {result}")
        
        if result is None:
            return jsonify({'error': 'Calculation error: result is invalid'})

        if temp < -7:
            green_result = 0.62*result
            jb_result = 0.73*result
            slicer_result = 0.8*result
        elif -7 <= temp <= -3:
            green_result = 0.55*result
            jb_result = 0.6*result
            blue_result = 0.85*result
            slicer_result = 0.82*result
        else:
            green_result = 0.48*result
            jb_result = 0.6*result
            slicer_result = 0.85*result

        return jsonify({
                        'normal_salt': round(result, 2),
                        'green': round(green_result, 2),
                        'jet_blue': round(jb_result, 2),
                        'blue': round(blue_result, 2) if blue_result!= 0 else 'N/A',
                        'slicer': round(slicer_result, 2)
                        })
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching Hossain data: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/weather/ice')
def get_ice(): 
    n = request.args.get('n', default=10, type=int)
    try:
        data = get_weather_records(n) # n - number of records to fetch
        if not data:
            return jsonify({'error': 'No data found'}), 404
        probability = predict(data)

        probability = round(probability*100, 2)

        response = {
            "probability": probability,
            "timestamp": data[0][5],
            "sequence_length": len(data)
        }
        return jsonify(response)

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching ice data: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run()