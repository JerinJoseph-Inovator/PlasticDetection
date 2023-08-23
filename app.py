from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from ultralytics import YOLO
import shutil
import os, random

import cv2
import numpy as np
from scipy.spatial import distance_matrix
from collections import defaultdict
import heapq

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
    

    model = YOLO("./best.pt")  
    # Run batched inference on a list of images
    results = model(source=imgz, save=True, save_txt=True) 

    # Specify the folder paths for images, YOLO annotation files, and output annotated images
    image_folder = r"./runs/detect/predict"
    annotation_folder = r"./runs/detect/predict/labels"
    output_folder = r"./runs/detect/"

    # Process images, find the shortest path with all centroids visited, and save annotated images with the path
    process_images(image_folder, annotation_folder, output_folder)

    #||||||||||||||||||||||
    # Geotagged the images
    # Spliting the Link string to get file name and creating file path
    trimmedImgz = imgz.split("/")[-1]
    imgz1 = f'./{trimmedImgz}'
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

    resultPath = f'runs/detect/{imgz.split("/")[-1]}'
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
        "public_url": blob.public_url
    }

def read_yolo_annotations(annotation_path):
    annotations = []
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            annotations.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return annotations

def calculate_distance_matrix(centroids):
    distance_mat = distance_matrix(centroids, centroids)
    return distance_mat

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def dijkstra_shortest_path(graph, start, end, unvisited_nodes):
    queue = [(0, start)]
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    previous = {}

    while queue:
        distance, current = heapq.heappop(queue)

        if distance > distances[current]:
            continue

        if current == end:
            break

        for neighbor, weight in graph[current]:
            if neighbor not in unvisited_nodes:
                continue
            new_distance = distance + weight
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))
                previous[neighbor] = current

    path = []
    while current != start:
        path.insert(0, current)
        current = previous[current]
    path.insert(0, start)

    return path, distances[end]

def find_shortest_path(centroids):
    num_centroids = len(centroids)
    distance_mat = calculate_distance_matrix(centroids)
    
    max_distance = -1
    start_point = None
    end_point = None

    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            dist = distance_mat[i][j]
            if dist > max_distance:
                max_distance = dist
                start_point = i
                end_point = j

    graph = defaultdict(list)
    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            weight = distance_mat[i][j]
            graph[i].append((j, weight))
            graph[j].append((i, weight))

    unvisited_nodes = set(range(num_centroids))
    unvisited_nodes.remove(start_point)
    unvisited_nodes.remove(end_point)

    total_distance = 0
    shortest_path = [start_point]

    current_idx = start_point
    while unvisited_nodes:
        next_idx = None
        min_distance = float('inf')

        for neighbor, weight in graph[current_idx]:
            if neighbor in unvisited_nodes and weight < min_distance:
                next_idx = neighbor
                min_distance = weight

        if next_idx is not None:
            shortest_path.append(next_idx)
            unvisited_nodes.remove(next_idx)
            total_distance += min_distance
            current_idx = next_idx
        else:
            break

    shortest_path.append(end_point)
    total_distance += distance_mat[current_idx][end_point]

    return shortest_path

def draw_shortest_path(image_path, annotations, shortest_path, output_path):
    try:
        image = cv2.imread(image_path)

        if image is None:
            raise Exception(f"Image not loaded: {image_path}")

        height, width, _ = image.shape
        total_distance_pixels = 0

        for num, idx in enumerate(shortest_path[1:], start=1):
            x1, y1, _, _ = annotations[shortest_path[num - 1]][1:]
            x2, y2, _, _ = annotations[idx][1:]
            start_point = (int(x1 * width), int(y1 * height))
            end_point = (int(x2 * width), int(y2 * height))
            thickness = 5  # Thickness of the line
            cv2.line(image, start_point, end_point, (255, 255, 255), thickness)
            distance_pixels = calculate_distance(start_point, end_point)
            total_distance_pixels += distance_pixels
            text = f"{distance_pixels:.2f} pixels"
            text_position = ((start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2)
            cv2.putText(image, str(num), text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (204, 255, 102), 3)
            cv2.putText(image, text, (text_position[0], text_position[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        start_idx = shortest_path[0]
        end_idx = shortest_path[-1]
        start_x, start_y, _, _ = annotations[start_idx][1:]
        end_x, end_y, _, _ = annotations[end_idx][1:]
        start_center = (int(start_x * width), int(start_y * height))
        end_center = (int(end_x * width), int(end_y * height))

        cv2.circle(image, start_center, 15, (0, 255, 0), -1)
        cv2.circle(image, end_center, 15, (255, 255, 255), -1)

        cv2.putText(image, "Start", (start_center[0] - 30, start_center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(image, "End", (end_center[0] - 30, end_center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        cv2.imwrite(output_path, image)
        print("Annotated image with thick white lines, distances, and marked start and end points saved:", output_path)
        print("Total pixel distance:", total_distance_pixels, "pixels")
        print("Total distance in meters:", total_distance_pixels * 1.28 * 0.01, "meters")

        return total_distance_pixels

    except Exception as e:
        print("An error occurred:", str(e))
        return 0

def process_images(image_folder, annotation_folder, output_folder):
    print("I am inside process_image")
    try:
        for image_filename in os.listdir(image_folder):
            if image_filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(image_folder, image_filename)
                annotation_filename = image_filename.replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".png", ".txt").replace(".bmp", ".txt")
                annotation_path = os.path.join(annotation_folder, annotation_filename)
                if os.path.isfile(annotation_path):
                    annotations = read_yolo_annotations(annotation_path)
                    centroids = np.array([[x, y] for _, x, y, _, _ in annotations])
                    shortest_path = find_shortest_path(centroids)
                    output_path = os.path.join(output_folder, image_filename)
                    draw_shortest_path(image_path, annotations, shortest_path, output_path)
                    print("Shortest path with all centroids visited saved:", output_path)

    except Exception as e:
        print("An error occurred:", str(e))

# https://storage.googleapis.com/reva2-aca6e.appspot.com/image.jpg

#im.close()
