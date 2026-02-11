import pandas as pd
import numpy as np
import requests

DATE_COL = "Date"

def fetch_weather_data_openmeteo(start_date,end_date,out_csv):
    """
    Fetch historical daily weather data for major US hub cities
    """

    start_date = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_date).strftime("%Y-%m-%d")

    cities = {
        "New York": (40.7128, -74.0060),
        "Los Angeles": (34.0522, -118.2437),
        "Chicago": (41.8781, -87.6298),
        "Atlanta": (33.7490, -84.3880),
        "Dallas": (32.7767, -96.7970),
        "Denver": (39.7392, -104.9903),
        "Miami": (25.7617, -80.1918),
        "Seattle": (47.6062, -122.3321),
        "Phoenix": (33.4484, -112.0740),
        "San Francisco": (37.7749, -122.4194),
    }

    url = "https://archive-api.open-meteo.com/v1/archive"

    daily_vars = [
        "wind_gusts_10m_max",
        "snowfall_sum",
        "precipitation_sum",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
    ]

    all_rows = []

    for city_name, (lat, lon) in cities.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(daily_vars),
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
        }

        print(f"Fetching weather data for {city_name}")
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            daily = data.get("daily", {})
            times = daily.get("time", None)

            gust = daily.get("wind_gusts_10m_max", [None] * len(times))
            snow = daily.get("snowfall_sum", [0] * len(times))
            precip = daily.get("precipitation_sum", [0] * len(times))
            temp_mean = daily.get("temperature_2m_mean", [None] * len(times))
            temp_max = daily.get("temperature_2m_max", [None] * len(times))
            temp_min = daily.get("temperature_2m_min", [None] * len(times))

            for i, t in enumerate(times):
                all_rows.append({
                    "date": pd.to_datetime(t),
                    "city": city_name,
                    "wind_gust_mph": np.nan if gust[i] is None else float(gust[i]),
                    "snow_in": 0.0 if snow[i] is None else float(snow[i]),
                    "precip_in": 0.0 if precip[i] is None else float(precip[i]),
                    "temp_mean": np.nan if temp_mean[i] is None else float(temp_mean[i]),
                    "temp_max": np.nan if temp_max[i] is None else float(temp_max[i]),
                    "temp_min": np.nan if temp_min[i] is None else float(temp_min[i]),
                })

        except Exception as e:
            print(f"Error fetching weather for {city_name}: {e}")
            continue

    weather_city = pd.DataFrame(all_rows)
    if weather_city.empty:
        print("Warning: No weather data fetched!")
        return pd.DataFrame()

    # create bad weather columns
    weather_city["is_high_wind"] = (weather_city["wind_gust_mph"] >= 45.0).astype(int)
    weather_city["is_heavy_snow"] = (weather_city["snow_in"] >= 3.0).astype(int)
    weather_city["is_heavy_rain"] = (weather_city["precip_in"] >= 1.0).astype(int)
    weather_city["is_extreme_cold"] = (weather_city["temp_min"] <= 20.0).astype(int)
    weather_city["is_extreme_heat"] = (weather_city["temp_max"] >= 95.0).astype(int)

    # Aggregate all cities data
    daily = (
        weather_city.groupby("date")
        .agg(
            n_cities=("city", "nunique"),
            
            # National temp (mean, min, max)
            weather_temp_mean=("temp_mean", "mean"),
            weather_temp_max=("temp_max", "max"),
            weather_temp_min=("temp_min", "min"),
            
            # Max single-city disruption
            weather_max_wind_gust=("wind_gust_mph", "max"),
            weather_max_snow=("snow_in", "max"),
            weather_max_precip=("precip_in", "max"),
            
            # Percentage of cities affected (0.0 to 1.0)
            weather_pct_high_wind=("is_high_wind", "mean"),
            weather_pct_heavy_snow=("is_heavy_snow", "mean"),
            weather_pct_heavy_rain=("is_heavy_rain", "mean"),
            weather_pct_extreme_cold=("is_extreme_cold", "mean"),
            weather_pct_extreme_heat=("is_extreme_heat", "mean"),
        )
        .reset_index()
    )

    # Create severe weather column
    daily["weather_is_severe"] = (
        (daily["weather_pct_high_wind"] >= 0.3) |
        (daily["weather_max_wind_gust"] >= 60.0) |
        (daily["weather_pct_heavy_snow"] >= 0.2) |
        (daily["weather_max_snow"] >= 8.0) |
        (daily["weather_pct_extreme_cold"] >= 0.4)
    ).astype(int)

    # Create weather disruption score which combines multiple factors into a 0-100 disruption score
    daily["weather_disruption_score"] = (
        daily["weather_pct_high_wind"] * 30 +
        daily["weather_pct_heavy_snow"] * 25 +
        daily["weather_pct_heavy_rain"] * 15 +
        daily["weather_pct_extreme_cold"] * 20 +
        daily["weather_pct_extreme_heat"] * 10
    ).clip(0, 100)

    # Select output features
    out = daily[[
        "date",
        
        # Binary flag
        "weather_is_severe",
        
        # Continuous score (0-100)
        "weather_disruption_score",
        
        # Individual percentages
        "weather_pct_high_wind",
        "weather_pct_heavy_snow",
        "weather_pct_extreme_cold",
        
        # Max values
        "weather_max_wind_gust",
        "weather_max_snow",
        
        # Temperature
        "weather_temp_mean",
    ]].copy()

    if out_csv:
        out.to_csv(out_csv, index=False)
        print(f"\nSaved: {out_csv}")

    return out


if __name__ == "__main__":
    # Load dataframes to determine date range
    df_train = pd.read_csv("tsa_train.csv", parse_dates=[DATE_COL])
    df_test = pd.read_csv("tsa_test.csv", parse_dates=[DATE_COL])

    start_date = min(df_train[DATE_COL].min(), df_test[DATE_COL].min())
    end_date = max(df_train[DATE_COL].max(), df_test[DATE_COL].max())

    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Total days: {(end_date - start_date).days}")

    print("FETCHING WEATHER DATA")

    weather_df = fetch_weather_data_openmeteo(
        start_date,
        end_date,
        out_csv="weather_data.csv",
    )

    print("First 10 rows:")
    print(weather_df.head(10))
    
    print("Last 10 rows:")
    print(weather_df.tail(10))