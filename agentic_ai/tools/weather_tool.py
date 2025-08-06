import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base_tool import BaseCustomTool, ToolParameter
from config import Config


class WeatherTool(BaseCustomTool):
    """Weather information tool using OpenWeatherMap API"""
    
    tool_name = "weather_info"
    tool_description = "Get current weather information and forecasts for any location"
    parameters = [
        ToolParameter(
            name="location",
            type="string",
            description="The city name, state, and/or country (e.g., 'New York, NY, USA' or 'London, UK')",
            required=True
        ),
        ToolParameter(
            name="forecast_days",
            type="integer",
            description="Number of forecast days to include (0 for current weather only, max 5)",
            required=False,
            default=0
        ),
        ToolParameter(
            name="units",
            type="string",
            description="Temperature units: 'metric' (Celsius), 'imperial' (Fahrenheit), or 'kelvin'",
            required=False,
            default="metric"
        )
    ]
    
    def __init__(self):
        super().__init__()
        self.api_key = Config.WEATHER_API_KEY
        self.base_url = Config.WEATHER_BASE_URL
        self.session = requests.Session()
    
    def _get_coordinates(self, location: str) -> Optional[Dict[str, float]]:
        """Get latitude and longitude for a location using geocoding API"""
        if not self.api_key:
            return None
        
        try:
            geocode_url = f"http://api.openweathermap.org/geo/1.0/direct"
            params = {
                'q': location,
                'limit': 1,
                'appid': self.api_key
            }
            
            response = self.session.get(geocode_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                return {
                    'lat': data[0]['lat'],
                    'lon': data[0]['lon'],
                    'name': data[0]['name'],
                    'country': data[0]['country'],
                    'state': data[0].get('state', '')
                }
            
            return None
            
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None
    
    def _get_current_weather(self, lat: float, lon: float, units: str = "metric") -> Dict[str, Any]:
        """Get current weather data"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': units
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information
            weather_info = {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'].title(),
                'main': data['weather'][0]['main'],
                'icon': data['weather'][0]['icon'],
                'visibility': data.get('visibility', 0) / 1000,  # Convert to km
                'uv_index': None,  # Not available in current weather API
                'wind': {
                    'speed': data['wind']['speed'],
                    'direction': data['wind'].get('deg', 0)
                },
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat()
            }
            
            # Add temperature unit
            unit_symbol = '°C' if units == 'metric' else '°F' if units == 'imperial' else 'K'
            weather_info['unit'] = unit_symbol
            
            return weather_info
            
        except Exception as e:
            return {'error': f"Error fetching current weather: {str(e)}"}
    
    def _get_forecast(self, lat: float, lon: float, days: int, units: str = "metric") -> List[Dict[str, Any]]:
        """Get weather forecast data"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': units,
                'cnt': min(days * 8, 40)  # 8 forecasts per day (every 3 hours), max 40
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            forecasts = []
            unit_symbol = '°C' if units == 'metric' else '°F' if units == 'imperial' else 'K'
            
            for item in data['list']:
                forecast = {
                    'datetime': datetime.fromtimestamp(item['dt']).isoformat(),
                    'temperature': item['main']['temp'],
                    'feels_like': item['main']['feels_like'],
                    'humidity': item['main']['humidity'],
                    'pressure': item['main']['pressure'],
                    'description': item['weather'][0]['description'].title(),
                    'main': item['weather'][0]['main'],
                    'icon': item['weather'][0]['icon'],
                    'wind': {
                        'speed': item['wind']['speed'],
                        'direction': item['wind'].get('deg', 0)
                    },
                    'precipitation': item.get('rain', {}).get('3h', 0) + item.get('snow', {}).get('3h', 0),
                    'unit': unit_symbol
                }
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            return [{'error': f"Error fetching forecast: {str(e)}"}]
    
    def _format_weather_summary(self, location_info: Dict, current: Dict, forecast: List = None) -> str:
        """Format weather information into a readable summary"""
        location_name = f"{location_info['name']}, {location_info['country']}"
        if location_info.get('state'):
            location_name = f"{location_info['name']}, {location_info['state']}, {location_info['country']}"
        
        summary = f"Weather for {location_name}:\n\n"
        
        if 'error' not in current:
            summary += f"Current: {current['temperature']}{current['unit']} - {current['description']}\n"
            summary += f"Feels like: {current['feels_like']}{current['unit']}\n"
            summary += f"Humidity: {current['humidity']}%\n"
            summary += f"Wind: {current['wind']['speed']} m/s\n"
        else:
            summary += f"Current weather: {current['error']}\n"
        
        if forecast and len(forecast) > 0 and 'error' not in forecast[0]:
            summary += f"\nForecast:\n"
            current_date = None
            for item in forecast[:24]:  # Limit to 24 items for readability
                forecast_date = datetime.fromisoformat(item['datetime']).date()
                if forecast_date != current_date:
                    summary += f"\n{forecast_date.strftime('%Y-%m-%d')}:\n"
                    current_date = forecast_date
                
                time_str = datetime.fromisoformat(item['datetime']).strftime('%H:%M')
                summary += f"  {time_str}: {item['temperature']}{item['unit']} - {item['description']}\n"
        
        return summary
    
    def _run(self, location: str, forecast_days: int = 0, units: str = "metric") -> Dict[str, Any]:
        """Execute weather information retrieval"""
        try:
            params = self.validate_parameters({
                "location": location,
                "forecast_days": forecast_days,
                "units": units
            })
            
            if not self.api_key:
                return {
                    "success": False,
                    "error": "Weather API key not configured. Please set WEATHER_API_KEY in your environment.",
                    "location": params["location"]
                }
            
            # Get coordinates for the location
            location_info = self._get_coordinates(params["location"])
            if not location_info:
                return {
                    "success": False,
                    "error": f"Location '{params['location']}' not found",
                    "location": params["location"]
                }
            
            # Get current weather
            current_weather = self._get_current_weather(
                location_info['lat'], 
                location_info['lon'], 
                params["units"]
            )
            
            result = {
                "success": True,
                "location": {
                    "name": location_info['name'],
                    "country": location_info['country'],
                    "state": location_info.get('state', ''),
                    "coordinates": {
                        "lat": location_info['lat'],
                        "lon": location_info['lon']
                    }
                },
                "current_weather": current_weather,
                "units": params["units"]
            }
            
            # Get forecast if requested
            if params["forecast_days"] > 0:
                forecast_data = self._get_forecast(
                    location_info['lat'],
                    location_info['lon'],
                    params["forecast_days"],
                    params["units"]
                )
                result["forecast"] = forecast_data
                result["forecast_days"] = params["forecast_days"]
            
            # Add formatted summary
            result["summary"] = self._format_weather_summary(
                location_info, 
                current_weather, 
                result.get("forecast")
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "location": location
            }
    
    async def _arun(self, location: str, forecast_days: int = 0, units: str = "metric") -> Dict[str, Any]:
        """Async version of weather information retrieval"""
        return self._run(location, forecast_days, units)
    
    def get_weather_alerts(self, location: str) -> Dict[str, Any]:
        """Get weather alerts for a location (requires One Call API subscription)"""
        # This would require the One Call API which needs a subscription
        # For now, return a placeholder
        return {
            "success": False,
            "error": "Weather alerts require One Call API subscription",
            "location": location
        }