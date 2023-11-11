import requests

# URL of your Flask API endpoint
url = 'http://localhost:5000/upload'  # Replace with your API's URL

# Path to the image file you want to upload
image_path = './test_images/20200617_185301b_contrast.jpg'  # Replace with the actual image file path

# Send a POST request with the image
try:
    with open(image_path, 'rb') as image_file:
        response = requests.post(url, files={'image': image_file})

    # Check the response from the server
    if response.status_code == 200:
        result = response.json()
        print("API Response:", result)
    else:
        print("API Request Failed with Status Code:", response.status_code)
except Exception as e:
    print("An error occurred:", str(e))
