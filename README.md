
# Weather Chatbot

This is a Python-based chatbot that provides weather forecasts by interacting with external APIs such as **OpenMeteo** and **Nominatim**. The chatbot uses Mistral AI for natural language processing and can provide weather information based on user inputs for specific locations and dates.

## Requirements

The script uses several Python packages and external APIs. These include:

- `openmeteo_requests` for weather data.
- `requests_cache` for caching API responses.
- `pandas` for data handling.
- `geopy` to interact with Nominatim for geolocation.
- `langchain_mistralai` to interact with Mistral AI.

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/jseltmann/WeatherChatbot.git
cd weather_chatbot
cd weather_chatbot
```

### 2. Install the required packages:

Make sure you have Python installed, then use the following command to install all the necessary dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Obtain a Mistral API Key

To use the Mistral AI model for natural language processing, you'll need to get an API key:

1. Visit [Mistral AI](https://mistral.ai) and sign up for an account.
2. Once you have an account, generate an API key from your account dashboard.

### 4. Set the Mistral API Key as an Environment Variable

You will need to store the API key in an environment variable named `MISTRAL_API_KEY`. You can do this by adding the following line to your terminal session or your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export MISTRAL_API_KEY="your_mistral_api_key_here"
```

Alternatively, you will be prompted to enter your Mistral API key when running the chatbot if the environment variable is not set.

## Services Used

### OpenMeteo

The chatbot retrieves weather forecasts using the [**OpenMeteo API**](https://open-meteo.com/), a free weather service that provides detailed daily weather information, including temperature, precipitation, wind speed, and gusts for the next 7 days.

### Nominatim

[**Nominatim**](https://nominatim.org/) is used for geolocation services. It takes a location (such as a city name) and converts it to latitude and longitude, which are required to query the weather forecast from OpenMeteo.

## How It Works

### 1. Starting the Chatbot

Run the script using:

```bash
python weather_chatbot.py
```

The chatbot will start a conversation where you can ask about the weather for specific places and dates. For example:

```
You: What’s the weather in Paris on Friday?
```

The chatbot will respond with weather information retrieved via OpenMeteo.

### 2. Bot Interaction Flow

- The chatbot takes user input, which includes the **location** and **days** for which weather is requested (you can use day names or specific dates).
- It converts the location into geographical coordinates using **Nominatim**.
- The weather data is then retrieved from **OpenMeteo** for up to 7 days.
- The chatbot uses **Mistral AI** to handle the conversation and integrate the weather forecast into a human-readable format.

### 3. Commands

- Type `exit` to quit the chatbot.
- The bot will remember the previous location if not provided in subsequent queries, allowing follow-up weather requests for the same place.

## Example Interaction

```bash
You: What’s the weather in Berlin tomorrow?
Bot: The weather forecast for Berlin tomorrow (Tuesday) shows a high of 18°C with 40% chance of rain.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
