import os
import gzip
import numpy as np
import h5py
import pickle
import sklearn
from sklearn.decomposition import PCA
import open3d

        
def save_dataset(dataset_dict, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(dataset_dict, f, protocol=2)

def save_dataset_and_compress(dataset_dict, name):
    with gzip.GzipFile(name + '.pgz', 'w') as f:
        pickle.dump(dataset_dict, f, protocol=2)
        
def load_dataset(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_pickle_file(pickle_file_name):
    with open(pickle_file_name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def load_compressed_pickle_file(pickle_file_name):
    with gzip.open(pickle_file_name, 'rb') as f:
        return pickle.load(f)
    
def load_pickle_file_with_label(pickle_file_name, compressed=False):
    if compressed:
        dataset = load_compressed_pickle_file(pickle_file_name)
    else:
        dataset = load_pickle_file(pickle_file_name)

    return (dataset['data'], dataset['labels'])
  
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,...]
    shuffled_labels = labels[permutation]
    
    return shuffled_dataset, shuffled_labels

def pca_compress(pointcloud, n_components=3):
    pca = PCA(n_components)
    pca.fit(pointcloud)

    return pca

def normalize_pointcloud(pointcloud):
    # Subtract mean of points from all the points to centre the pointcloud at 0,0,0
    pointcloud_xyz = pointcloud[:, 0:3]
    number_of_points = pointcloud_xyz.shape[0]
    centre = np.sum(pointcloud, axis=0) / number_of_points
    pointcloud_xyz[:, 0] -= centre[0]
    pointcloud_xyz[:, 1] -= centre[1]
    pointcloud_xyz[:, 2] -= centre[2]
    principal_components = pca_compress(pointcloud_xyz).components_
    squared_length_principal_components = np.multiply(principal_components, principal_components)
    length_principal_components = np.sqrt(np.sum(squared_length_principal_components, axis=1))

    # Calculate rotation matrix
    R = principal_components
    R[0, :] = R[0, :] / length_principal_components[0]
    R[1, :] = R[1, :] / length_principal_components[1]
    R[2, :] = R[2, :] / length_principal_components[2]

    # rotate the pointcloud
    if pointcloud.shape[1] > 3:  # if colour is part of the pointcloud
        normalized_pointcloud = np.hstack([R.dot(pointcloud_xyz.T).T, pointcloud[:, 3:]])
    else:
        normalized_pointcloud = R.dot(pointcloud_xyz.T).T

    return normalized_pointcloud
  
def extract_pcd(pcd_file, 
                num_points=2048,
                color=True, 
                downsample_cloud=True,
                pad_cloud=True,
                normalize_cloud=True
                ):
    
    cloud = open3d.io.read_point_cloud(pcd_file)
    
    if np.asarray(cloud.points).shape[0] > 0:
        #downsample cloud until the size < num_points
        if downsample_cloud:
            # start with big voxel size for cloud with many points
            if np.asarray(cloud.points).shape[0] > 10000:
                cloud = cloud.voxel_down_sample(voxel_size = 0.01)
            elif np.asarray(cloud.points).shape[0] > 5000:
                cloud = cloud.voxel_down_sample(voxel_size = 0.0040)

            cloud_iteration = 0
            voxel_filter = 0.0020
            filter_adjusment = 0.0001
            while (np.asarray(cloud.points).shape[0] > num_points):
                cloud = cloud.voxel_down_sample(voxel_size = voxel_filter)
                if cloud_iteration > 5:
                    voxel_filter = voxel_filter + filter_adjusment
                    cloud_iteration = 0
                cloud_iteration += 1

        xyzrgb = np.hstack((np.asarray(cloud.points), np.asarray(cloud.colors)))
        #pad cloud until its size == num_points
        if pad_cloud:
            while xyzrgb.shape[0] < num_points:
                rand_idx = np.random.randint(xyzrgb.shape[0])
                xyzrgb = np.vstack([xyzrgb, xyzrgb[rand_idx]])

        #normalize_cloud
        if normalize_cloud:
            normalize_pointcloud(xyzrgb)
            
    else:
        return None
            
    return xyzrgb