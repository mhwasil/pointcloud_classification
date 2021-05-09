from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
import sys
import provider
import yaml

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    #print(out_str)

def get_gmm(points, n_gaussians, NUM_POINT, type='grid', variance=0.05, n_scales=3, D=3):
    """
    Compute weights, means and covariances for a gmm with two possible types 'grid' (2D/3D) and 'learned'

    :param points: num_points_per_model*nummodels X 3 - xyz coordinates
    :param n_gaussians: scalar of number of gaussians /  number of subdivisions for grid type
    :param NUM_POINT: number of points per model
    :param type: 'grid' / 'leared' toggle between gmm methods
    :param variance: gaussian variance for grid type gmm
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    if type == 'grid':
        #Generate gaussians on a grid - supports only axis equal subdivisions
        if n_gaussians >= 32:
            print('Warning: You have set a very large number of subdivisions.')
        if not(isinstance(n_gaussians, list)):
            if D == 2:
                gmm = get_2d_grid_gmm(subdivisions=[n_gaussians, n_gaussians], variance=variance)
            elif D == 3:
                gmm = get_3d_grid_gmm(subdivisions=[n_gaussians, n_gaussians, n_gaussians], variance=variance)
            else:
                ValueError('Wrong dimension. This supports either D=2 or D=3')

    elif type == 'learn':
        #learn the gmm from given data and save to gmm.p file, if already learned then load it from existing gmm.p file for speed
        if isinstance(n_gaussians, list):
            raise  ValueError('Wrong number of gaussians: non-grid value must be a scalar')
        print("Computing GMM from data - this may take a while...")
        info_str = "g" + str(n_gaussians) + "_N" + str(len(points)) + "_M" + str(len(points) / NUM_POINT)
        gmm_dir = "gmms"
        if not os.path.exists(gmm_dir):
            os.mkdir(gmm_dir)
        filename = gmm_dir + "/gmm_" + info_str + ".p"
        if os.path.isfile(filename):
            gmm = pickle.load(open(filename, "rb"))
        else:
            gmm = get_learned_gmm(points, n_gaussians, covariance_type='diag')
            pickle.dump(gmm, open( filename, "wb"))
    else:
        ValueError('Wrong type of GMM [grid/learn]')

    return gmm


def get_learned_gmm(points, n_gaussians, covariance_type='diag'):
    """
    Learn weights, means and covariances for a gmm based on input data using sklearn EM algorithm

    :param points: num_points_per_model*nummodels X 3 - xyz coordinates
    :param n_gaussians: scalar of number of gaussians /  3 element list of number of subdivisions for grid type
    :param covariance_type: Specify the type of covariance mmatrix : 'diag', 'full','tied', 'spherical' (Note that the Fisher Vector method relies on diagonal covariance matrix)
        See sklearn documentation : http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    gmm = GaussianMixture(n_components = n_gaussians, covariance_type=covariance_type)
    gmm.fit(points.astype(np.float64))
    return gmm


def get_3d_grid_gmm(subdivisions=[5,5,5], variance=0.04):
    """
    Compute the weight, mean and covariance of a gmm placed on a 3D grid
    :param subdivisions: 2 element list of number of subdivisions of the 3D space in each axes to form the grid
    :param variance: scalar for spherical gmm.p
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    # n_gaussians = reduce(lambda x, y: x*y,subdivisions)
    n_gaussians = np.prod(np.array(subdivisions))
    step = [1.0/(subdivisions[0]),  1.0/(subdivisions[1]),  1.0/(subdivisions[2])]

    means = np.mgrid[ step[0]-1: 1.0-step[0]: complex(0, subdivisions[0]),
                      step[1]-1: 1.0-step[1]: complex(0, subdivisions[1]),
                      step[2]-1: 1.0-step[2]: complex(0, subdivisions[2])]
    means = np.reshape(means, [3, -1]).T
    covariances = variance*np.ones_like(means)
    weights = (1.0/n_gaussians)*np.ones(n_gaussians)
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'diag')
    return gmm


def get_2d_grid_gmm(subdivisions=[5, 5], variance=0.04):
    """
    Compute the weight, mean and covariance of a 2D gmm placed on a 2D grid

    :param subdivisions: 2 element list of number of subdivisions of the 2D space in each axes to form the grid
    :param variance: scalar for spherical gmm.p
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    # n_gaussians = reduce(lambda x, y: x*y,subdivisions)
    n_gaussians = np.prod(np.array(subdivisions))
    step = [1.0/(subdivisions[0]),  1.0/(subdivisions[1])]

    means = np.mgrid[step[0]-1: 1.0-step[0]: complex(0, subdivisions[0]),
            step[1]-1: 1.0-step[1]: complex(0, subdivisions[1])]
    means = np.reshape(means, [2,-1]).T
    covariances = variance*np.ones_like(means)
    weights = (1.0/n_gaussians)*np.ones(n_gaussians)
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'diag')
    return gmm


def get_fisher_vectors(points,gmm, normalization=True):
    """
    Compute the fisher vector representation of a point cloud or a batch of point clouds

    :param points: n_points x 3 / B x n_points x 3
    :param gmm: sklearn MixtureModel class containing the gmm.p parameters.p
    :return: fisher vector representation for a single point cloud or a batch of point clouds
    """

    if len(points.shape) == 2:
        # single point cloud
        fv = fisher_vector(points, gmm, normalization=normalization)
    else:
        # Batch of  point clouds
        fv = []
        n_models = points.shape[0]
        for i in range(n_models):
            fv.append(fisher_vector(points[i], gmm, normalization=True))
        fv = np.array(fv)
    return fv


def fisher_vector(xx, gmm, normalization=True):
    """
    Computes the Fisher vector on a set of descriptors.
    code from : https://gist.github.cnsom/danoneata/9927923
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors

    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.

    Returns
    -------
    fv: array_like, shape (K + 128 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.

    Reference
    ---------
    Sanchez, J., Perronnin, F., Mensink, T., & Verbeek, J. (2013).
    Image classification with the fisher vector: Theory and practice. International journal of computer vision, 105(64), 222-245.
    https://hal.inria.fr/hal-00830491/file/journal.pdf

    """
    xx = np.atleast_2d(xx)
    n_points = xx.shape[0]
    D = gmm.means_.shape[1]
    tiled_weights = np.tile(np.expand_dims(gmm.weights_, axis=-1), [1, D])

    #start = time.time()
    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK
    #mid = time.time()
    #print("Computing the probabilities took ", str(mid-start))
    #Compute Derivatives

    # Compute the sufficient statistics of descriptors.
    s0 = np.sum(Q, 0)[:, np.newaxis] / n_points
    s1 = np.dot(Q.T, xx) / n_points
    s2 = np.dot(Q.T, xx ** 2) / n_points

    d_pi = (s0.squeeze() - n_points * gmm.weights_) / np.sqrt(gmm.weights_)
    d_mu = (s1 - gmm.means_ * s0 ) / np.sqrt(tiled_weights*gmm.covariances_)
    d_sigma = (
        + s2
        - 2 * s1 * gmm.means_
        + s0 * gmm.means_ ** 2
        - s0 * gmm.covariances_
        ) / (np.sqrt(2*tiled_weights)*gmm.covariances_)

    #Power normaliation
    alpha = 0.5
    d_pi = np.sign(d_pi) * np.power(np.absolute(d_pi),alpha)
    d_mu = np.sign(d_mu) * np.power(np.absolute(d_mu), alpha)
    d_sigma = np.sign(d_sigma) * np.power(np.absolute(d_sigma), alpha)

    if normalization == True:
        d_pi = normalize(d_pi[:,np.newaxis], axis=0).ravel()
        d_mu = normalize(d_mu, axis=0)
        d_sigma = normalize(d_sigma, axis=0)
    # Merge derivatives into a vector.

    #print("comnputing the derivatives took ", str(time.time()-mid))

    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def fisher_vector_per_point( xx, gmm):
    """
    see notes for above function - performs operations per point

    :param xx: array_like, shape (N, D) or (D, )- The set of descriptors
    :param gmm: instance of sklearn mixture.GMM object - Gauassian mixture model of the descriptors.
    :return: fv_per_point : fisher vector per point (derivative by w, derivative by mu, derivative by sigma)
    """
    xx = np.atleast_2d(xx)
    n_points = xx.shape[0]
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]

    sig2 = np.array([gmm.covariances_.T[0, :], gmm.covariances_.T[1, :], gmm.covariances_.T[2,:]]).T
    sig2_tiled = np.tile(np.expand_dims(sig2, axis=0), [n_points, 1, 1])

    # Compute derivativees per point and then sum.
    Q = gmm.predict_proba(xx)  # NxK
    tiled_weights = np.tile(np.expand_dims(gmm.weights_, axis=-1), [1, D])
    sqrt_w = np.sqrt(tiled_weights)

    d_pi = (Q - np.tile(np.expand_dims(gmm.weights_, 0), [n_points, 1])) / np.sqrt(np.tile(np.expand_dims(gmm.weights_, 0), [n_points, 1]))
    x_mu = np.tile( np.expand_dims(xx, axis=2), [1, 1, n_gaussians]) - np.tile(np.expand_dims(gmm.means_.T, axis=0), [n_points, 1, 1])
    x_mu = np.swapaxes(x_mu, 1, 2)
    d_mu = (np.tile(np.expand_dims(Q, -1), D) * x_mu) / (np.sqrt(sig2_tiled) * sqrt_w)

    d_sigma =   np.tile(np.expand_dims(Q, -1), 3)*((np.power(x_mu,2)/sig2_tiled)-1)/(np.sqrt(2)*sqrt_w)

    fv_per_point = (d_pi, d_mu, d_sigma)
    return fv_per_point


def l2_normalize(v, dim=1):
    """
    Normalize a vector along a dimension

    :param v: a vector or matrix to normalize
    :param dim: the dimension along which to normalize
    :return: normalized v along dim
    """
    norm = np.linalg.norm(v, axis=dim)
    if norm.all() == 0:
       return v
    return v / norm
  
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

    # find first three principal components and rotate pointcloud so the first principal
    # component is aligned along the x-axis, 2nd principal component along y-axis and the
    # third along the z-axis
    principal_components = pca_compress(pointcloud_xyz).components_
    #print (principal_components.shape)
    squared_length_principal_components = np.multiply(principal_components, principal_components)
    #print (squared_length_principal_components.shape)
    length_principal_components = np.sqrt(np.sum(squared_length_principal_components, axis=1))
    #print ("length",length_principal_components.shape)

    # Calculate rotation matrix
    R = principal_components
    R[0, :] = R[0, :] / length_principal_components[0]
    R[1, :] = R[1, :] / length_principal_components[1]
    R[2, :] = R[2, :] / length_principal_components[2]

    #print ("R",R.shape)

    # rotate the pointcloud
    if pointcloud.shape[1] > 3:  # if colour is part of the pointcloud
        normalized_pointcloud = np.hstack([R.dot(pointcloud_xyz.T).T, pointcloud[:, 3:]])
    else:
        normalized_pointcloud = R.dot(pointcloud_xyz.T).T

    return normalized_pointcloud
  
def normalize_batch_pointcloud(batch_pointcloud):
    normalized_batch_pointcloud = []
    for pointcloud in batch_pointcloud:
        normalized_batch_pointcloud.append(normalize_pointcloud(pointcloud))
        
    return np.asarray(normalized_batch_pointcloud)

# centers a point cloud around the origin and scales it to fit a unit sphere
# INPUT: cloud - Nx3 point cloud
# OUTPUT: cloud - Nx3 scaled point cloud
def scale_to_unit_sphere(batch_pointcloud, normalize=False):
    scaled_pointcloud = []
    for points in batch_pointcloud:
        xyz = points[0:3]
        if normalize:
            centroid = np.mean(xyz, axis=0)
            xyz = xyz - centroid
        scale = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        if scale > 0.0:
            xyz = xyz / scale
            
        points[0:3] = xyz
            
        scaled_pointcloud.append(points)
      
    return np.asarray(scaled_pointcloud)
    #return scaled_pointcloud

# Pad points for variable lenght input
def pad_batch_pointcloud(batch_pointcloud, max_points=2048):
    padded_batch_pointcloud = []
    for cloud in batch_pointcloud:
        padded_cloud = cloud.tolist()
        while (len(padded_cloud) < max_points):
            padded_cloud.append([0,0,0])
        padded_batch_pointcloud.append(padded_cloud)
            
    return np.asarray(padded_batch_pointcloud)

def randomize_batch(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,...]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

def get_yaml_contents(yaml_file):
    if os.path.isfile(yaml_file):
        configs = {}
        with open(yaml_file, 'r') as infile:
            configs = yaml.safe_load(infile)
    else:
        print("Configfile not found")
        return None
      
    return configs
