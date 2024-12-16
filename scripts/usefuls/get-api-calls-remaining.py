import requests

# API URL
api_url = "https://api.football-data-api.com/test-call?key=dcfab3f4b36acc7031c6ddaa1212e1c35d750da0a8f3257c771c93df954b374c"

try:
    # Perform the API call
    response = requests.get(api_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        
        # Check if the API call was successful
        if data.get("success"):
            # Extract and display the remaining API calls
            request_remaining = data.get("metadata", {}).get("request_remaining")
            print(f"API Calls Remaining: {request_remaining}")
        else:
            print("API call failed. Check the API response for details.")
    else:
        print(f"Failed to make the API call. HTTP Status Code: {response.status_code}")

except Exception as e:
    print(f"An error occurred: {str(e)}")
