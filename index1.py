from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import math
import logging
import os


app = Flask(__name__, static_folder='../frontend/build', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*"}})
load_dotenv()

# API Keys and Endpoint
ENDPOINT = "https://rt.ambientweather.net/v1/devices?"
API_KEY = '80d0cbf141844d2e8afdd8dd4caf9d6979393c605d85477386ceff64899bb067'
APP_KEY = 'a784dc6788c74457b7d77bbe095f1b9f9488b233772d4d92bab211b9ea3bac85'
# API_KEY = os.getenv('API_KEY')
# APP_KEY = os.getenv('APP_KEY')

logging.basicConfig(level=logging.DEBUG)

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

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/api/weather')
def get_weather():
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
    try:
        # Use the stored temperature data
        sample_data = {
            'snow_height': 0.25,
            'snow_density': 300,
            'salt_concentration': 0.23,
            'salt_temp': -21,
            'pave_temp': f_to_c(last_tempf),
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
        return jsonify({'Hossain_2014_Result': result if result is not None else "Calculation error"})
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching Hossain data: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()