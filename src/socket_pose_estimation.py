import numpy as np
import time
import concurrent.futures
import threading
import traceback
import datetime
#from extract_teat_params import get_pixel_coords_of_ends
from itertools import chain
import pyransac3d as pyrsc
import multiprocessing
import cv2
import statistics as stat

# intrinsic camera matrix - RGB camera
K_RGB = np.array([[456.5170009681582, 0.0, 332.8146709049914],[0.0, 458.1027893118239, 243.1950147690160],[0.0, 0.0, 1.0]])
        
# intrinsic camera matrix - Depth camera
K_depth = np.array([[476.65203857421875, 0.0, 320.2263488769531],[0.0, 476.65203857421875, 193.44775390625],[0.0, 0.0, 1.0]])

# translation matrix 
T1 = np.array([[1,0,0, 0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,1]])

def extract_pose(masks, point_cloud):
  centers = []
  with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
    # Start the threads and collect results
    results = {executor.submit(compute_centers, masks[index], point_cloud) for index in range(len(masks))}
    for future in concurrent.futures.as_completed(results):
      try:
          socket_center = future.result()
          if (socket_center != None):
            centers.append(socket_center)
      except Exception as exc:
          print('Thread generated an exception: %s' % exc)
          traceback.print_exc()

    vertical_vector = []
    horizontal_vector = []
    normal_vector = []

    if (len(centers) == 2):
        vertical_vector = compute_socket_vector(centers)
        vertical_vector = vertical_vector/np.linalg.norm(vertical_vector)
        normal_vector = compute_normal(masks, point_cloud)
        normal_vector = normal_vector/np.linalg.norm(normal_vector)*0.1                 # vectors are 10 cm long
        horizontal_vector = np.cross(vertical_vector,normal_vector)
        horizontal_vector = horizontal_vector/np.linalg.norm(horizontal_vector)*0.1     # vectors are 10 cm long
        vertical_vector = np.cross(normal_vector, horizontal_vector)
        vertical_vector = vertical_vector/np.linalg.norm(vertical_vector)*0.1           # vectors are 10 cm long

        #quat = get_quaternion_from_vector(normal_vector)
        rotm = get_rotation_from_vectors(horizontal_vector,vertical_vector, normal_vector)
        pose = [centers[1], rotm]
        return [centers, normal_vector, horizontal_vector, vertical_vector, pose]
    else:
        return [None, None, None, None, None]

def compute_centers(mask, point_cloud):
    start = time.time()
    points = []
    vertices = []

    # get indexes of points in mask, these are also the u,v coordinates of the mask pixels in the color image
    idx = np.where(mask == True)
    rows, row_idx, counts = np.unique(idx[0], return_index=True, return_counts=True)

    all_rows = len(rows)
    used_rows = int(all_rows) # use 100% of rows
  
    # we use a 3d point for each 2d point on the "edge" of the mask
    mask_edge_points = np.ones((2 * 2* used_rows + 1, 3)) # with this we're setting all z coordinates to 1
    # mask_edge_points = np.ones((2 * len(rows) +1, 3))
    mask_edge_points[-1] = [0, 0, 0]

    # print(np.shape(mask_edge_points))
    #for i in range(len(rows)):
    counter = 0
    for i in chain(range(used_rows),range(all_rows - used_rows,all_rows)):
        row = rows[i]
        # row, first column, last column
        mask_edge_points[2*counter][0] = idx[1][row_idx[i]] # x coordinate in pixels
        mask_edge_points[2*counter][1] = row # y coordinate in pixels
        mask_edge_points[2*counter+1][0] = idx[1][row_idx[i] + counts[i] -1] # x coordinate in pixels
        mask_edge_points[2*counter+1][1] = row # y coordinate in pixels
        counter+=1

    # from the 2D points at the edge of the mask we want to calculate rays in 3D
    # cv::Point3d ray;
    # ray.x = (uv_rect.x - cx - Tx) / fx;
    # ray.y = (uv_rect.y - cy - Ty) / fy;
    # ray.z = 1.0;
    # print(mask_edge_points)
    # TODO: the formula above comes from projectPixelTo3dRay in ROS, but the
    # resulting vector has not modulo 1, still it seems correct

    #FIXME: we're using rays as vertices (setting depth=1) and building a frustum with them.
    # We should probably use max_depth from the pointcloud to do the right computation
    rays = (mask_edge_points - [K_RGB[0,2], K_RGB[1,2] , 0]) / [K_RGB[0,0], K_RGB[1,1], 1] - [T1[0,3], T1[1,3], 0]
    rays[-1] = [0, 0, 0]
  
    # we multiply the rays for min and max distance to get 3D vertices
    # print(rays)
    points_subset, _ = extract_pc_in_box3d(np.reshape(point_cloud, (-1,3)), rays)

    # remove NaNs
    points_subset = points_subset[~np.isnan(points_subset).any(axis=1)]
    #print(min(points_subset[0]), max(points_subset[0]), min(points_subset[1]), max(points_subset[1]), min(points_subset[2]), max(points_subset[2]))
    stime = time.time()
    print("PC subset extraction time: %s", format(stime - start))

    # we use clustering to get rid of outliers and noise
    # points_subset = cluster_largest(points_subset[:,0:3], eps=0.003, show=False, debug=False)
    # avg = 0
    # for i in range(len(points_subset[:, 0])):
    #     avg += points_subset[i, 2]
    # avg = avg / len(points_subset[:, 0])
    # indexes = []
    # for i in range(len(points_subset[:, 0])):
    #     if (points_subset[i, 2] >= avg+0.1):
    #         indexes.insert(0, i)
    # points_subset = np.delete(points_subset, indexes, 0)

    # calculate center of each mask
    sumX_pose = 0
    sumY_pose = 0
    sumZ_pose = 0
    for i in range(len(points_subset[:,0])):
        sumX_pose += points_subset[i,0]
        sumY_pose += points_subset[i,1]
        sumZ_pose += points_subset[i,2]
    avgZ = sumZ_pose/len(points_subset[:,0])

    avgPx = 0
    avgPy = 0
    print("idx: ", np.shape(idx))
    for i in range(len(idx[0])):
        avgPx += idx[1][i]
        avgPy += idx[0][i]
    avgPx = avgPx/len(idx[0])
    avgPy = avgPy/len(idx[0])

    fovWidth = 2 * np.tan(35.025*np.pi/180) * avgZ
    fovHeight = 2 * np.tan(27.65*np.pi/180) * avgZ
    avgX = avgPx/640*fovWidth-fovWidth/2
    avgY = avgPy/480*fovHeight-fovHeight/2

    return avgX, avgY, avgZ


def compute_normal(masks, point_cloud):
    start = time.time()
    vertices = []
    x = []
    y = []
    z = []

    for mask in masks:
        # get indexes of points in mask, these are also the u,v coordinates of the mask pixels in the color image
        idx = np.where(mask == True)
        rows, row_idx, counts = np.unique(idx[0], return_index=True, return_counts=True)

        all_rows = len(rows)
        used_rows = int(all_rows * 100 / 100)  # use 100% of rows

        # we use a 3d point for each 2d point on the "edge" of the mask
        mask_edge_points = np.ones((2 * 2 * used_rows + 1, 3))  # with this we're setting all z coordinates to 1
        # mask_edge_points = np.ones((2 * len(rows) +1, 3))
        mask_edge_points[-1] = [0, 0, 0]
        # for i in range(len(rows)):
        counter = 0
        for i in chain(range(used_rows), range(all_rows - used_rows, all_rows)):
            row = rows[i]
            # row, first column, last column
            mask_edge_points[2 * counter][0] = idx[1][row_idx[i]]  # x coordinate in pixels
            mask_edge_points[2 * counter][1] = row  # y coordinate in pixels
            mask_edge_points[2 * counter + 1][0] = idx[1][row_idx[i] + counts[i] - 1]  # x coordinate in pixels
            mask_edge_points[2 * counter + 1][1] = row  # y coordinate in pixels
            counter += 1

        # FIXME: we're using rays as vertices (setting depth=1) and building a frustum with them.
        # We should probably use max_depth from the pointcloud to do the right computation
        rays = (mask_edge_points - [K_RGB[0, 2], K_RGB[1, 2], 0]) / [K_RGB[0, 0], K_RGB[1, 1], 1] - [T1[0, 3], T1[1, 3], 0]
        rays[-1] = [0, 0, 0]

        # we multiply the rays for min and max distance to get 3D vertices
        #print(rays)
        points_subset, _ = extract_pc_in_box3d(np.reshape(point_cloud, (-1, 3)), rays)

        # remove NaNs
        points_subset = points_subset[~np.isnan(points_subset).any(axis=1)]

        x = np.concatenate([x,points_subset[:,0]])
        y = np.concatenate([y,points_subset[:,1]])
        z = np.concatenate([z,points_subset[:,2]])

    # combine all points in 1 array
    points = np.vstack((x,y,z)).T
    print(points)
    # avg = 0
    # for i in range(len(points[:,0])):
    #     avg += points[i,2]
    # avg = avg/len(points[:,0])
    # indexes = []
    # for i in range(len(points[:,0])):
    #     if (points[i,2] >= avg-0.005):
    #         indexes.insert(0,i)
    # points = np.delete(points,indexes,0)
    plane = pyrsc.Plane()
    best_eq, best_inliers = plane.fit(points, 0.003)
    normal_vector = [best_eq[0],best_eq[1],best_eq[2]]

    normal_vector = calculate_orientation(normal_vector)

    return normal_vector

def compute_socket_vector(centers):
    point1 = centers[0];
    point2 = centers[1];

    socket_vector = [point1[0]-point2[0], point1[1]-point2[1], point1[2]-point2[2]]
    return socket_vector
    

def calculate_orientation(vector):
    if (vector[2] < 0):
      vector = [-vector[0], -vector[1], -vector[2]]
    return vector

def get_quaternion_from_vector(vector):
    rot_vector = np.cross([1, 0, 0], vector)
    rot_vector = rot_vector / np.linalg.norm(rot_vector)
    angle = np.arccos(np.dot([1, 0, 0], vector))
    quat = [rot_vector[0] * np.sin(angle/2), rot_vector[1] * np.sin(angle/2), rot_vector[2] * np.sin(angle/2), np.cos(angle/2)]
    return quat

def get_rotation_from_vectors(horizontal_vector, vertical_vector, normal_vector):
    vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    horizontal_vector = horizontal_vector / np.linalg.norm(horizontal_vector)
    rotation_mat = [horizontal_vector, vertical_vector, normal_vector]
    return rotation_mat

def get_rotation_from_vector(vector):
    rot_vector = np.cross([0, 0, 1], vector)
    rot_vector = rot_vector / np.linalg.norm(rot_vector)
    angle = np.arccos(np.dot([1, 0, 0], vector))
    quat = [rot_vector[0] * np.sin(angle/2), rot_vector[1] * np.sin(angle/2), rot_vector[2] * np.sin(angle/2), np.cos(angle/2)]
    return quat


def cluster_largest(points, eps=0.003, show=False, debug=False):
  from sklearn.cluster import DBSCAN
  from sklearn import metrics
  stime = time.time()
  
  # Compute DBSCAN
  # NOTE: clustering parameters depend on points representation (in meters / pixel)
  # They have to be adjusted accordingly
  db = DBSCAN(eps=eps, min_samples=8).fit(points)
  core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True
  labels = db.labels_

  # Number of clusters in labels, ignoring noise if present.
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  n_noise_ = list(labels).count(-1)

  print('Estimated number of clusters: %d' % n_clusters_)
  print('Estimated number of noise points: %d' % n_noise_)
  etime = time.time()
  print("Cluster computation time: %s" % (etime - stime))
  #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(points, labels))
  
  if n_clusters_ > 0:
    unique_labels = set(labels)
    clusters = []
    for k in unique_labels:
      class_member_mask = (labels == k)
      xyz = points[class_member_mask & core_samples_mask]
      if np.shape(xyz)[0] > 0:
        clusters.append([k, # index of cluster
                        np.shape(xyz)[0], # size
                        np.average(xyz[:, 0]), # avg_x
                        np.average(xyz[:, 1]), # avg_y
                        np.average(xyz[:, 2]), # avg depth
                        np.std(xyz[:, 0]), # stdev_x
                        np.std(xyz[:, 1]), # stdev_y
                        np.std(xyz[:, 2]), # stdev_z
                      ])
    clusters = np.array(clusters)
  #  print(clusters)
    
    # we take the largest
    if len(clusters) > 0:

      largest_idx = np.argmax(clusters[:,1], axis=0) # cluster_size
      closest_idx = np.argmin(clusters[:,4], axis=0) # depth
      if (largest_idx == closest_idx):
        teat_idx = clusters[largest_idx][0] # index of cluster
      else:
        print("Closest cluster is not the largest: discarding")
        if debug:
          #use the multiprocessing module to perform the plotting activity in another process:
          plot_job = multiprocessing.Process(target=plot_clusters,args=(points, labels, core_samples_mask, largest_idx, n_clusters_))
          plot_job.start()
        return []
    #  print(teat_idx)

      if show:
        #use the multiprocessing module to perform the plotting activity in another process:
        plot_job = multiprocessing.Process(target=plot_clusters,args=(points, labels, core_samples_mask, teat_idx, n_clusters_))
        plot_job.start()

      # return only points of closest cluster
      class_member_mask = (labels == teat_idx)
      xyz = points[class_member_mask & core_samples_mask]
      return xyz

# #############################################################################
  # Plot result
# #############################################################################
def plot_clusters(points, labels, core_samples_mask, teat_idx, n_clusters_):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D

  # Black removed and is used for noise instead.
  unique_labels = set(labels)
  colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  plt.ylabel("row (y)") 
  plt.xlabel("column (x)")
  ax.set_zlabel("depth (z)")
  for k, col in zip(unique_labels, colors):
      if k == -1:
          # Black used for noise.
          col = [0, 0, 0, 1]

      class_member_mask = (labels == k)
      xyz = points[class_member_mask & core_samples_mask]
      ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], marker='o', color=tuple(col))

      xyz_nc = points[class_member_mask & ~core_samples_mask]
      ax.scatter(xyz_nc[:, 0], xyz_nc[:, 1], xyz_nc[:,2], marker='x', color=tuple(col))

  plt.title('Estimated number of clusters: %d' % n_clusters_)
  ax.view_init(elev=-75., azim=-90)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  class_member_mask = (labels == teat_idx)
  xyz = points[class_member_mask & core_samples_mask]
  ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], marker='o')

  xyz_nc = points[class_member_mask & ~core_samples_mask]
  ax.scatter(xyz_nc[:, 0], xyz_nc[:, 1], xyz_nc[:,2], marker='x')
  plt.title('Teat cluster points')
  plt.ylabel("row (y)") 
  plt.xlabel("column (x)")
  ax.set_zlabel("depth (z)")
  ax.view_init(elev=-75., azim=-90)
  plt.show()

def plot_points(points):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = -points[:][1]
    y = -points[:][0]
    ax.scatter(x, y)
    plt.ylabel("row (y)")
    plt.xlabel("column (x)")
    plt.show()

def plot_o3d(points, voxel_size=0.001, radius=0.04):
  import open3d as o3d
  pcd = o3d.geometry.PointCloud()  
  pcd.points = o3d.utility.Vector3dVector(points)
  pcd_no_outliers, _ = pcd.remove_radius_outlier(50, radius=radius)
  pcd = pcd_no_outliers.voxel_down_sample(voxel_size=voxel_size) #0.005 #5
  #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=20)) # 0.04 #20
  plot_job = multiprocessing.Process(target=o3d.visualization.draw_geometries, args=([pcd,],), kwargs={'point_show_normal': False})
  plot_job.start()

def view_pcl(points):
    import pptk    
    v = pptk.viewer(points)


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds

def add_mask(image, masks, color=(1.0, 0, 1.0), alpha=0.5):
    """Apply mask to image.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    # Copy color pixels from the original color image where mask is set
    if masks.shape[0] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(masks, axis=0) >= 1)
        splash = np.zeros(np.shape(image), dtype=np.uint8)
        for c in range(3):
            splash[:, :, c] = np.where(mask == 1,
                 image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                 image[:, :, c])        
    else:
        splash = image
    return splash

def extractMeanDistance(mask, point_cloud):

    return meanDistance
        

if __name__ == "__main__":

    # # Display images
    # cv2.imshow('RGB Image', imgRGB)  
    
    #masks = [np.zeros((480,640))]
    #masks[0][233,336] = 1
    
    teat_pose = extract_pose(masks, point_cloud)

    # Add teat tip on rgb
    # add marker for each teat tip
    for pose in teat_pose:
        # transform 3d point to rgb
        point = np.array([pose[0][0],pose[0][1],pose[0][2],1])
        pointCorr = np.matmul(T1,point)
        px,py,lambda1 = np.matmul(K_RGB,pointCorr[0:3])
        if ~np.isnan(px) and ~np.isnan(px):
            px = int(px/lambda1)
            py = int(py/lambda1)
            cv2.drawMarker(imgRGB,(px,py),(255,0,0),thickness=2)
    # # Display images
    cv2.namedWindow("RGB Image")
    cv2.imshow('RGB Image', imgRGB)  

    # add mask
    splash = add_mask(imgRGB, masks)
    cv2.namedWindow("Masked Image")
    cv2.imshow('Masked Image', splash)
    cv2.waitKey(0)
    
