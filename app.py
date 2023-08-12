from flask import Flask, request
import json
from ultralytics import YOLO
import shutil
import PIL.Image
import PIL.ExifTags
import firebase_admin  
from firebase_admin import credentials
from firebase_admin import storage


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    args = request.args
    imgz, uuid = args.values()
    return process(imgz,uuid)
if __name__ == '__main__':
    app.run()


def process(imgz, uuid):
    cred = credentials.Certificate("./key.json")
    firebase_admin.initialize_app(cred, {'storageBucket': 'plastic-detection-598e8.appspot.com'})
    # to read and predict plastic in image
    # pretrained YOLOv8 model
   
    model = YOLO("./best.pt")  
    # Run batched inference on a list of images
    results = model(source=imgz, save=True) 

    #||||||||||||||||||||||
    # Geotagged the images
    # Spliting the Link string to get file name and creating file path
    trimmedImgz = imgz.split("/")[-1]
    imgz1 = f'./{trimmedImgz}'
    temp = PIL.Image.open(imgz1)
    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in temp._getexif().items()
        if k in PIL.ExifTags.TAGS
    }
    north = exif["GPSInfo"][2]
    east = exif["GPSInfo"][4]
    lat = ((((north[0] * 60) + north[1]) * 60) + north[2]) / 60 / 60
    lon = ((((east[0] * 60) + east[1]) * 60) + east[2]) / 60 / 60
    lat, lon = float(lat), float(lon)
    # RESULT
    geo_tag = f"https://www.google.com/maps/place/10%C2%B053'45.1%22N+106%C2%B041'38.3%22E/@{lat},{lon}"
    resultPath = f'runs/detect/predict/{imgz.split("/")[-1]}'
    # Result Back To Firebase Storage
    bucket = storage.bucket()
    blob = bucket.blob(f"{uuid}/results/{trimmedImgz}")
    blob.upload_from_filename(resultPath)
    blob.make_public()
    shutil.rmtree("./runs", ignore_errors=True)


    return {
        "geo_tag": geo_tag,
        "public_url": blob.public_url
    }

# https://storage.googleapis.com/reva2-aca6e.appspot.com/image.jpg