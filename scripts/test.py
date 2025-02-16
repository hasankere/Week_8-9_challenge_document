import requests

# Define the URL of the Flask app's /predict endpoint
url = 'http://127.0.0.1:5000/predict'

# Prepare the input data (Replace with your actual feature names and values)
data = {
    "feature1": 10,  # Example feature value (you should replace this with real data)
    "feature2": 20,  # Replace with actual feature name and value
    "feature3": 30   # Replace with actual feature name and value
}

# Send the POST request to the Flask /predict endpoint
response = requests.post(url, json=data)

# Check the response and print it
if response.status_code == 200:
    # Successful response from Flask (Prediction result)
    print("Prediction result:", response.json())
else:
    # If the request fails, print the error message
    print("Error:", response.json())
