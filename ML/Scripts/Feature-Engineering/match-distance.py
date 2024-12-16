from pymongo import MongoClient
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError
import logging
import time
import json
import os
import re

# ------------------------
# 1. Configure Logging
# ------------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs during troubleshooting
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------------
# 2. Configuration
# ------------------------
# MongoDB connection settings
MONGO_HOST = 'localhost'
MONGO_PORT = 27017

# Database and collection names
DATABASE_NAME = "footballDB"               # Source database
MATCHES_COLLECTION = "matches"            # Source collection
RESULTS_COLLECTION = "modelv5_results"    # Target collection

# Competition IDs to filter matches
COMPETITION_IDS = [246, 12, 11, 10, 9, 161, 1625, 2012, 4759, 6135, 7704, 9660]  # Replace with your actual competition IDs

# Geocoder initialization with increased timeout
geolocator = Nominatim(user_agent="stadium_distance_calculator", timeout=10)

# Cache file path
CACHE_FILE = "stadium_cache.json"

# ------------------------
# 3. Load or Initialize Cache
# ------------------------
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as cache_file:
            stadium_cache = json.load(cache_file)
        logger.info(f"Loaded stadium cache from '{CACHE_FILE}'.")
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading cache file '{CACHE_FILE}': {e}")
        stadium_cache = {}
else:
    stadium_cache = {}
    logger.info(f"No existing cache found. Starting with an empty cache.")

def save_cache():
    """
    Save the current stadium cache to the cache file.
    """
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as cache_file:
            json.dump(stadium_cache, cache_file, ensure_ascii=False, indent=4)
        logger.info(f"Saved updated stadium cache to '{CACHE_FILE}'.")
    except IOError as e:
        logger.error(f"Error saving cache to '{CACHE_FILE}': {e}")

# ------------------------
# 4. Helper Functions
# ------------------------
def extract_city(stadium_name):
    """
    Extract the city from the stadium name.
    Handles 'Stadium Name (City, Region)', 'Stadium Name (City)', and 'Stadium Name, City' formats.
    """
    # First, attempt to extract from parentheses
    match = re.search(r'\(([^,]+)(?:,\s*[^)]+)?\)', stadium_name)
    if match:
        city = match.group(1).strip()
        return city
    
    # If no parentheses, attempt to extract after comma
    match = re.search(r',\s*([^,]+)$', stadium_name)
    if match:
        city = match.group(1).strip()
        return city
    
    # If no comma, attempt to extract after last space
    match = re.search(r'\b([A-Za-z\s]+)$', stadium_name)
    if match:
        city = match.group(1).strip()
        return city
    
    return None

def fetch_matches(collection, competition_ids):
    """
    Fetch matches from the MongoDB collection based on competition_ids.
    """
    query = {"competition_id": {"$in": list(competition_ids)}}
    projection = {
        "stadium_name": 1,
        "awayID": 1,
        "homeID": 1,
        "date": 1,    # Ensure 'date' field is included for sorting
        "id": 1       # Ensure 'id' field is included
    }
    matches_cursor = collection.find(query, projection)
    
    # Count matches - handling deprecation of count()
    try:
        match_count = matches_cursor.count()
    except AttributeError:
        match_count = collection.count_documents(query)
    
    logger.info(f"Fetched {match_count} matches from the database.")
    return matches_cursor

def get_previous_match(collection, away_id):
    """
    Find the most recent match where the away_id was the homeID.
    Assumes there is a 'date' field to sort by in descending order.
    """
    query = {"homeID": away_id}
    projection = {"stadium_name": 1, "date": 1}  # Adjust fields as necessary
    previous_match = collection.find_one(query, projection, sort=[("date", -1)])
    return previous_match

def geocode_stadium_or_city(stadium_name, retries=3):
    """
    Geocode the stadium name to get its latitude and longitude.
    If geocoding fails, attempt to geocode the city name extracted from stadium_name.
    Implements caching to minimize geocoding requests.
    Includes a retry mechanism for transient errors.
    """
    # Check cache for stadium
    if stadium_name in stadium_cache:
        logger.debug(f"Using cached coordinates for stadium: {stadium_name}")
        return tuple(stadium_cache[stadium_name])
    
    # Append country for better accuracy
    stadium_query = f"{stadium_name}, England"
    
    for attempt in range(retries):
        try:
            location = geolocator.geocode(stadium_query)
            if location:
                coords = (location.latitude, location.longitude)
                stadium_cache[stadium_name] = list(coords)  # Store as list for JSON compatibility
                logger.info(f"Geocoded '{stadium_name}' to coordinates: {coords}")
                # Respect Nominatim's usage policy
                time.sleep(1)  # 1-second delay
                return coords
            else:
                logger.warning(f"Geocoding failed for stadium: {stadium_query}. Attempting to geocode the city.")
                # Extract city
                city = extract_city(stadium_name)
                if city:
                    city_query = f"{city}, England"
                    # Check if city is already in cache
                    if city_query in stadium_cache:
                        logger.debug(f"Using cached coordinates for city: {city_query}")
                        return tuple(stadium_cache[city_query])
                    # Attempt to geocode city
                    location = geolocator.geocode(city_query)
                    if location:
                        coords = (location.latitude, location.longitude)
                        stadium_cache[city_query] = list(coords)  # Store as list for JSON compatibility
                        logger.info(f"Geocoded city '{city}' to coordinates: {coords}")
                        # Respect Nominatim's usage policy
                        time.sleep(1)  # 1-second delay
                        return coords
                    else:
                        logger.warning(f"Geocoding failed for city: {city_query}")
                        return None
                else:
                    logger.warning(f"Could not extract city from stadium name: {stadium_name}")
                    return None
        except GeocoderServiceError as e:
            logger.error(f"Geocoding service error for stadium '{stadium_query}': {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying geocoding for stadium '{stadium_query}' (Attempt {attempt + 2}/{retries}) after delay.")
                time.sleep(2)  # Wait before retrying
            else:
                logger.error(f"Max retries exceeded for stadium '{stadium_query}'. Attempting to geocode the city as fallback.")
                # Attempt to geocode the city as a final fallback
                city = extract_city(stadium_name)
                if city:
                    city_query = f"{city}, England"
                    if city_query in stadium_cache:
                        logger.debug(f"Using cached coordinates for city: {city_query}")
                        return tuple(stadium_cache[city_query])
                    try:
                        location = geolocator.geocode(city_query)
                        if location:
                            coords = (location.latitude, location.longitude)
                            stadium_cache[city_query] = list(coords)
                            logger.info(f"Geocoded city '{city}' to coordinates: {coords}")
                            time.sleep(1)  # Respect usage policy
                            return coords
                        else:
                            logger.warning(f"Geocoding failed for city: {city_query}")
                            return None
                    except GeocoderServiceError as e2:
                        logger.error(f"Geocoding service error for city '{city_query}': {e2}")
                        return None
                else:
                    logger.warning(f"Could not extract city from stadium name: {stadium_name}")
                    return None
        except Exception as e:
            logger.error(f"Unexpected error during geocoding for stadium '{stadium_query}': {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying geocoding for stadium '{stadium_query}' (Attempt {attempt + 2}/{retries}) after delay.")
                time.sleep(2)  # Wait before retrying
            else:
                logger.error(f"Max retries exceeded for stadium '{stadium_query}'. Attempting to geocode the city as fallback.")
                # Attempt to geocode the city as a final fallback
                city = extract_city(stadium_name)
                if city:
                    city_query = f"{city}, England"
                    if city_query in stadium_cache:
                        logger.debug(f"Using cached coordinates for city: {city_query}")
                        return tuple(stadium_cache[city_query])
                    try:
                        location = geolocator.geocode(city_query)
                        if location:
                            coords = (location.latitude, location.longitude)
                            stadium_cache[city_query] = list(coords)
                            logger.info(f"Geocoded city '{city}' to coordinates: {coords}")
                            time.sleep(1)  # Respect usage policy
                            return coords
                        else:
                            logger.warning(f"Geocoding failed for city: {city_query}")
                            return None
                    except GeocoderServiceError as e2:
                        logger.error(f"Geocoding service error for city '{city_query}': {e2}")
                        return None
                else:
                    logger.warning(f"Could not extract city from stadium name: {stadium_name}")
                    return None
    logger.error(f"Failed to geocode stadium '{stadium_query}' after {retries} attempts.")
    return None

def calculate_distance(loc1, loc2):
    """
    Calculate geodesic distance between two locations.
    loc1 and loc2 should be tuples of (latitude, longitude).
    """
    try:
        distance = geodesic(loc1, loc2).kilometers
        return distance
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return None

# ------------------------
# 5. Main Processing Function
# ------------------------
def main():
    # ------------------------
    # a. Connect to MongoDB
    # ------------------------
    try:
        client = MongoClient(MONGO_HOST, MONGO_PORT)
        db = client[DATABASE_NAME]
        matches_collection = db[MATCHES_COLLECTION]
        results_collection = db[RESULTS_COLLECTION]
        logger.info("Connected to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return  # Exit the script if connection fails
    
    # ------------------------
    # b. Fetch Matches
    # ------------------------
    matches = fetch_matches(matches_collection, COMPETITION_IDS)
    
    updated_count = 0
    not_found_count = 0
    
    # ------------------------
    # c. Process Each Match
    # ------------------------
    for match in matches:
        try:
            # Extract necessary fields
            home_stadium_name = match.get("stadium_name")
            away_id = match.get("awayID")
            match_id = match.get("id")  # Correctly retrieve 'id'
            
            if match_id is None:
                logger.warning(f"Match document missing 'id': {match}")
                not_found_count += 1
                continue
            
            if not home_stadium_name or not away_id:
                logger.warning(f"Match {match_id} missing home stadium name or awayID.")
                continue
            
            # Geocode home stadium or city
            home_coords = geocode_stadium_or_city(home_stadium_name)
            if not home_coords:
                logger.warning(f"Could not geocode home stadium '{home_stadium_name}' for match {match_id}.")
                continue
            
            # Get previous match where awayID was homeID
            previous_match = get_previous_match(matches_collection, away_id)
            if not previous_match:
                logger.warning(f"No previous match found where homeID is {away_id} for match {match_id}.")
                continue
            
            away_stadium_name = previous_match.get("stadium_name")
            if not away_stadium_name:
                logger.warning(f"Previous match {previous_match.get('_id')} missing stadium_name.")
                continue
            
            # Geocode away stadium or city
            away_coords = geocode_stadium_or_city(away_stadium_name)
            if not away_coords:
                logger.warning(f"Could not geocode away stadium '{away_stadium_name}' for match {match_id}.")
                continue
            
            # Calculate distance
            distance_km = calculate_distance(home_coords, away_coords)
            if distance_km is not None:
                # Update the corresponding document in 'modelv5_results'
                update_result = results_collection.update_one(
                    {"match_id": match_id},
                    {"$set": {"distance_km": round(distance_km, 2)}}
                )
                
                if update_result.matched_count > 0:
                    logger.info(f"Match {match_id}: Distance between '{home_stadium_name}' and '{away_stadium_name}' is {distance_km:.2f} KM")
                    updated_count += 1
                else:
                    logger.warning(f"No matching document found in '{RESULTS_COLLECTION}' for match_id: {match_id}")
                    not_found_count += 1
                
        except Exception as e:
            # If 'match_id' is missing or other errors occur
            match_id = match.get("id", "Unknown")
            logger.error(f"Error processing match {match_id}: {e}")
            continue
    
    # ------------------------
    # d. Final Logging
    # ------------------------
    logger.info(f"Successfully updated {updated_count} records in '{RESULTS_COLLECTION}'.")
    if not_found_count > 0:
        logger.warning(f"Could not find {not_found_count} matching documents in '{RESULTS_COLLECTION}'.")
    
    # ------------------------
    # e. Save Cache and Close Connection
    # ------------------------
    save_cache()
    
    # Close MongoDB connection
    client.close()
    logger.info("Closed MongoDB connection.")

# ------------------------
# 6. Execute Main Function
# ------------------------
if __name__ == "__main__":
    main()
