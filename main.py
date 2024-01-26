# name: Ali Akbar Ahrari

import numpy as np
from PIL import Image


def process_images():
    image_collection = []

    for img_num in range(1, 6):  # Processing five images
        img_path = f'images/usps_{img_num}.jpg'
        image = Image.open(img_path)
        img_array = np.array(image)

        small_images = [
            img_array[i:i + 16, j:j + 16].flatten()
            for i in range(0, img_array.shape[0], 16)
            for j in range(0, img_array.shape[1], 16)
        ]

        image_collection.extend(small_images)

    return np.array(image_collection)


def categorize_data(data, centers):
    num_data_points = data.shape[0]
    num_clusters = centers.shape[0]

    all_distances = np.zeros((num_data_points, num_clusters))

    for i in range(num_clusters):
        diff = data - centers[i]
        squared_distances = np.sum(diff ** 2, axis=1)
        all_distances[:, i] = squared_distances

    assigned_labels = np.argmin(all_distances, axis=1)
    return assigned_labels


def update_cluster_centers(data, labels, num_clusters):
    num_features = data.shape[1]
    new_centers = np.zeros((num_clusters, num_features))

    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]

        if len(cluster_indices) > 0:
            cluster_data = data[cluster_indices]
            new_centers[i] = np.sum(cluster_data, axis=0) / len(cluster_indices)

    return new_centers


def run_k_means_clustering(data, num_clusters, initial_centers, max_iterations=100):
    centers = initial_centers

    for step in range(max_iterations):
        print(f'Iteration: {step} | Cluster Count = {num_clusters}')

        assigned_labels = categorize_data(data, centers)
        centers = update_cluster_centers(data, assigned_labels, num_clusters)

    return centers, assigned_labels


if __name__ == '__main__':
    data = process_images()
    for k in [3, 4, 5, 6, 7]:
        c_initial = data[np.random.choice(len(data), k) ]
        centroids, label = run_k_means_clustering(data, k,c_initial)

        centroids = centroids.reshape((k, 16, 16))
        for i in range(k):
            image_data = np.uint8(centroids[i])
            image = Image.fromarray(image_data)
            image.save(f"centroids/{k}/centroid{i + 1}.png")
