from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from ultralytics import YOLO
import shutil
import os, random

from files1 import process_images

import PIL.Image
import PIL.ExifTags
import firebase_admin  
from firebase_admin import credentials
from firebase_admin import storage

app = Flask(__name__)
CORS(app)
cred = credentials.Certificate("./key.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'plastic-detection-598e8.appspot.com'})

@app.route('/', methods=['GET'])
def hello_world():
    try:
        # Attempt to get the value of the 'arg' parameter from the request
        args = request.args
        imgz, uuid = args.values()
        
        if imgz and uuid is None:
            # If 'arg' is not provided, raise an exception
            raise ValueError("No 'arg' parameter provided.")
        # Rest of your code here... 
        return jsonify(process(imgz,uuid))

    except ValueError as e:
        return f"Error: {e}", 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0')

    

def process(imgz, uuid):
    # For checking and Deleting runs folder
    delete_runs_folders()

    model = YOLO("./best.pt")  
    results = model(source=imgz, save=True, save_txt=True) 


    # Spliting the Link string to get file name and creating file path
    trimmedImgz = imgz.split("/")[-1]
    imgz1 = f'./{trimmedImgz}'
    label1 = trimmedImgz.replace(".jpg", ".txt")
    total_distance_meters, num_plastics, output_path = process_images()
    # Geotagged the images
    try:
        temp = PIL.Image.open(imgz1)
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in temp._getexif().items()
            if k in PIL.ExifTags.TAGS
        }

        if "GPSInfo" in exif:
            north = exif["GPSInfo"][2]
            east = exif["GPSInfo"][4]
            lat = ((((north[0] * 60) + north[1]) * 60) + north[2]) / 60 / 60
            lon = ((((east[0] * 60) + east[1]) * 60) + east[2]) / 60 / 60
            lat, lon = float(lat), float(lon)
            # RESULT
            geo_tag = f"https://www.google.com/maps/place/10%C2%B053'45.1%22N+106%C2%B041'38.3%22E/@{lat},{lon}"
        else:
            # No GPSInfo found in the image
            geo_tag = "No GPS information available"
    except Exception as e:
    # Handle any other exceptions that might occur during image processing
        geo_tag = f"Error: {str(e)}"
        # ||| END OF GEO TAGGING | STARTING TO STORE DATA IN FIREBASE DB

    # resultPath = f'runs/detect/{imgz.split("/")[-1]}'
    resultPath = rf"{output_path}" 
    
    # Result Back To Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f"{uuid}/results/{trimmedImgz}")
    if blob.exists():
            blob.delete()
    blob.upload_from_filename(resultPath)
    blob.make_public()

    
    shutil.rmtree("./runs", ignore_errors=True) 
    temp.close()
    os.remove(imgz1)
    

    return {
        "geo_tag": geo_tag,
        "public_url": blob.public_url,
        "num_plastic": num_plastics, 
        "Total_distance": total_distance_meters 
    }

def delete_runs_folders():
    current_directory = os.path.abspath(os.getcwd())
    
    # Delete the "runs" folder in the current directory
    current_runs_folder = os.path.join(current_directory, "runs")
    if os.path.exists(current_runs_folder) and os.path.isdir(current_runs_folder):
        try:
            shutil.rmtree(current_runs_folder)
            print(f"Deleted '{current_runs_folder}' folder in the current directory.")
        except OSError as e:
            print(f"Error deleting folder: {e}")

    # Delete ".png" and ".jpg" files in the current directory
    try:
        for item in os.listdir(current_directory):
            item_path = os.path.join(current_directory, item)
            if os.path.isfile(item_path) and (item.lower().endswith(".jpg") or item.lower().endswith(".png")):
                os.remove(item_path)
                print(f"Deleted file: {item}")
    except OSError as e:
        print(f"Error deleting files: {e}")



# https://storage.googleapis.com/reva2-aca6e.appspot.com/image.jpg

#im.close()
