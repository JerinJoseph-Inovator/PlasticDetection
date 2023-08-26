import cv2
import numpy as np
import os
from scipy.spatial import distance_matrix
from collections import defaultdict
import heapq

def read_yolo_annotations(annotation_path, plastic_class):
    annotations = []
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split(' ')
            class_idx = int(parts[0])
            if class_idx == plastic_class:
                annotations.append((class_idx, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
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

def draw_shortest_path(image_path, annotations, shortest_path, output_path, path_thickness=8):
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
            cv2.line(image, start_point, end_point, (255, 255, 255), path_thickness)
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

        start_text_size = cv2.getTextSize("Start", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        end_text_size = cv2.getTextSize("End", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]

        start_text_x = start_center[0] - start_text_size[0] // 2
        start_text_y = start_center[1] + start_text_size[1] + 5
        cv2.putText(image, "Start", (start_text_x, start_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        end_text_x = end_center[0] - end_text_size[0] // 2
        end_text_y = end_center[1] + end_text_size[1] + 5
        cv2.putText(image, "End", (end_text_x, end_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Draw rectangles around the text "Start" and "End"
        text_padding = 10
        cv2.rectangle(image,
                      (start_text_x - text_padding, start_text_y + text_padding),
                      (start_text_x + start_text_size[0] + text_padding, start_text_y - start_text_size[1] - text_padding),
                      (255, 255, 255), -1)
        cv2.putText(image, "Start", (start_text_x, start_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        cv2.rectangle(image,
                      (end_text_x - text_padding, end_text_y + text_padding),
                      (end_text_x + end_text_size[0] + text_padding, end_text_y - end_text_size[1] - text_padding),
                      (255, 255, 255), -1)
        cv2.putText(image, "End", (end_text_x, end_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        cv2.imwrite(output_path, image)
        print("Annotated image with white rectangles, distances, and marked start and end points saved:", output_path)
        print("Total pixel distance:", total_distance_pixels, "pixels")
        print("Total distance in meters:", total_distance_pixels * 1.28 / 100, "meters")

        return total_distance_pixels, total_distance_pixels * 1.28 / 100, len(shortest_path) - 2

    except Exception as e:
        print("An error occurred:", str(e))
        return 0, 0, 0

def process_images():

    input_folder = "./runs/detect/predict"
    output_folder = "./runs/detect"
    annotation_folder = "./runs/detect/predict/labels"
    plastic_class_index = 0
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [file for file in os.listdir(input_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        annotation_path = os.path.join(annotation_folder, os.path.splitext(image_file)[0] + ".txt")

        plastic_annotations = read_yolo_annotations(annotation_path, plastic_class_index)
        plastic_centroids = np.array([[x, y] for _, x, y, _, _ in plastic_annotations])
        shortest_path = find_shortest_path(plastic_centroids)

        output_path = os.path.join(output_folder, image_file.replace(".", "_plastic_annotated."))
        total_distance_pixels, total_distance_meters, num_plastics = draw_shortest_path(image_path, plastic_annotations, shortest_path, output_path)
        print("Annotated image saved:", output_path)
        print("Total distance in meters:", total_distance_meters, "meters")
        print("Number of plastics detected:", num_plastics)

        return total_distance_meters, num_plastics, output_path



