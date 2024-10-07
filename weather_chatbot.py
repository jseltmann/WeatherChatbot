import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from langchain_core.tools import tool, ToolException
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from geopy.geocoders import Nominatim
import datetime
import os
import getpass
import argparse
import time
import numpy as np

# Initialize tools and services
try:
    geolocator = Nominatim(user_agent="weather_chatbot")
except Exception as e:
    print("Could not reach geolocator service.")

try:
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
except Exception as e:
    print("Could not connect to OpenMeteo.")

def get_weather(latitude, longitude):
    """
    Retrieves weather data from OpenMeteo API for the specified latitude and longitude.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.

    Returns:
        pd.DataFrame: A DataFrame containing daily weather data, including temperature,
                      precipitation, wind speed, and wind gusts over the next 7 days.
    """
    today = datetime.datetime.today()
    start_date = today.strftime("%Y-%m-%d")
    end_day = today + datetime.timedelta(days=6)
    end_date = end_day.strftime("%Y-%m-%d")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": ["temperature_2m_max", "precipitation_sum", "precipitation_hours", 
                  "precipitation_probability_max", "wind_speed_10m_max", "wind_gusts_10m_max"],
        "timezone": "Europe/Berlin",
        "start_date": start_date,
        "end_date": end_date
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    daily = response.Daily()
    daily_temperature_2m_max = np.round(daily.Variables(0).ValuesAsNumpy())
    daily_precipitation_sum = np.round(daily.Variables(1).ValuesAsNumpy())
    daily_precipitation_hours = np.round(daily.Variables(2).ValuesAsNumpy())
    daily_precipitation_probability_max = daily.Variables(3).ValuesAsNumpy()
    daily_wind_speed_10m_max = np.round(daily.Variables(4).ValuesAsNumpy())
    daily_wind_gusts_10m_max = np.round(daily.Variables(5).ValuesAsNumpy())
    utc_offset = response.UtcOffsetSeconds()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time() + utc_offset, unit="s", utc=False),
            end=pd.to_datetime(daily.TimeEnd() + utc_offset, unit="s", utc=False),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        ),
        "temperature_2m_max": daily_temperature_2m_max,
        "precipitation_sum": daily_precipitation_sum,
        "precipitation_hours": daily_precipitation_hours,
        "precipitation_probability_max": daily_precipitation_probability_max,
        "wind_speed_10m_max": daily_wind_speed_10m_max,
        "wind_gusts_10m_max": daily_wind_gusts_10m_max
    }

    daily_dataframe = pd.DataFrame(data=daily_data)
    return daily_dataframe

@tool(parse_docstring=True)
def make_weather_call(place: str, days: list[str]):
    """
    Calls the OpenMeteo API to retrieve weather information for a specific location over certain days.

    Args:
        place: The location (city or region) to get the weather for.
        days: List of days for which to retrieve weather, 
              either in weekday names or date format (dd.mm.yyyy).

    Returns:
        str or dict: JSON string with the weather data or error messages if location is invalid or data could not be fetched.
    """
    try:
        location = geolocator.geocode(place)
    except Exception as e:
        return "The location could not be found due to an error with the Geolocation."
    if location is None:
        return "The given location could not be found."

    weekdays = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
    
    today = datetime.datetime.today()
    days_to_get_weather = []
    messages = []

    for day in days:
        day = day.lower()
        if day in weekdays:
            day_number = weekdays[day]
            day_relative = (day_number - today.weekday()) % 7
        else:
            date_obj = None
            for datestr in ["%d.%m.%Y", "%d.%m", "%Y-%m-%d"]:
                try:
                    date_obj = datetime.datetime.strptime(day, datestr)
                    break
                except ValueError:
                    continue
            
            if date_obj:
                delta = (date_obj - today).days
                if delta < 0:
                    messages.append(f"{day} is in the past.")
                    continue
                elif delta > 6:
                    messages.append(f"{day} is too far in the future.")
                    continue
                day_relative = delta
            else:
                continue

        days_to_get_weather.append(day_relative)

    if not days_to_get_weather:
        days_to_get_weather = range(7)

    if messages:
        return {"error": messages}

    try:
        weather_data = get_weather(location.latitude, location.longitude)
    except Exception as e:
        return "Could not retrieve weather data."

    filtered_df = weather_data.loc[days_to_get_weather].copy()
    filtered_df['day_of_week'] = pd.to_datetime(filtered_df['date']).dt.day_name()
    
    return filtered_df.to_json(orient='records', date_format='iso')

def run_chatbot():
    """
    Starts the interactive weather chatbot using Mistral AI and weather tool integration.
    Prompts the user for MISTRAL API key and interacts through a command-line interface.

    The chatbot fetches weather forecasts for user-provided locations and dates using OpenMeteo API.
    """
    if "MISTRAL_API_KEY" not in os.environ:
        os.environ["MISTRAL_API_KEY"] = getpass.getpass(prompt="Enter MISTRAL API Key: ")
    llm = ChatMistralAI(model="mistral-large-latest")

    llm_with_tools = llm.bind_tools([make_weather_call])
    llm_forced_to_use_tools = llm.bind_tools([make_weather_call], tool_choice='any')
    print("Starting the weather chatbot. Type 'exit' to quit.\n")

    # Loop for the interactive chatbot
    start = True
    previous_place = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if previous_place != "":
            user_input = f"Previously, I asked about {previous_place}. {user_input}"
        today = datetime.datetime.today()
        weekdays = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        current_weekday = weekdays[today.weekday()]
        current_date = today.strftime('%Y-%m-%d')

        system_message_base = "You are a meteorologist telling people about the weather forecast for the next seven days. Please use normal/simple language, like 'rain' instead of 'precipitation'."
        messages = [
            SystemMessage(content=f"{system_message_base} Today is {current_weekday}, {current_date}."),
            HumanMessage(content=user_input)
        ]

        try:
            if start:
                response = llm_forced_to_use_tools.invoke(messages)
                start = False
            else:
                response = llm_with_tools.invoke(messages)
        except Exception as e:
            print("Error: The LLM could not be reached. Please try again later.")
            continue
        messages.append(response)

        if len(response.tool_calls) > 0:
            time.sleep(1)
            for tool_call in response.tool_calls:

                selected_tool = {"make_weather_call": make_weather_call}[tool_call["name"].lower()]
                tool_msg = selected_tool.invoke(tool_call)
                messages.append(tool_msg)
                previous_place = tool_call['args']['place']

            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                print("Error: The LLM could not be reached. Please try again later.")
                continue
            messages.append(response)
        print("Bot: ", response.content)

if __name__ == "__main__":
    run_chatbot()