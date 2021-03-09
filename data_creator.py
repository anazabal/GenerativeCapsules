import numpy as np
import code
# code.interact(local=dict(globals(), **locals()))

#Templates for different objects
def create_square():
    return np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])
def create_triangle(): #return np.array([[-np.sqrt(3) / 2, -0.5], [0, 1], [np.sqrt(3) / 2, -0.5]])
    return np.array([[1, 2], [3, 1], [3, 3]])
def create_trapezoid():
    return np.array([[-0.5, 1], [0.5, 1], [1, -1], [-1, -1]])
def create_pentagon():
    return np.array([[-0.5, 0], [0, -1], [0.5, -1], [0.5, 0.5], [0, 1]])
def create_L_shape():
    return np.array([[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])
def create_plane():
    return np.array([[-2, -2], [-1, 0], [-1, 2], [-4, 2], [-1, 4], [-1, 6], [0, 8],
                                     [1, 6], [1, 4], [4, 2], [1, 2], [1, 0], [2, -2]])

# Helper function to create templates (arrays of 2D points) for each specified object in obj
def create_template(obj):
    template_list = ['square', 'triangle', 'trapezoid', 'pentagon', 'L_shape', 'plane']
    template_functions = {elem: 'create_' + elem for elem in template_list}
    if obj in template_list:
        return eval(template_functions[obj])()
    else:
        raise ValueError('Unimplemented object "' + obj + '"')

# This function creates the transformation matrix for a given 2D-point {list of 2x4 numpy arrays, number of vertices}
def expand_template(point):
    return [np.array([[1, 0, x, y], [0, 1, y, -x]]) for x, y in point]
# This function constraints the data to be in the [-1,1] 2D square
def constrain_image(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
# Add random noise to the points is specified
def noisy_observations(x, std_noise):
    return x + std_noise * np.random.randn(x.shape[0], 2)
# Transform templates such as they are cenetered and has a norm N_k
def normalize_template(obj):
    N_k = np.shape(obj)[0]
    obj = obj - np.mean(obj,0)
    obj = obj*np.sqrt(N_k/np.sum(obj**2))
    return obj

#This function generates an image given the expanded templates with a random affine transformation
def image_generation(templates, visible_objects, parameters):

    # Create template characteristic matrices, noised if necessary
    F_true = list(map(expand_template, templates))
    if parameters['noise_type'] == 'template':
        noisy_templates = [noisy_observations(template, parameters['std_noise']) for template in templates]
        F = list(map(expand_template, noisy_templates))
    else:
        F = F_true

    # Create transformed data points with random transformations
    y = np.random.randn(len(visible_objects), 4)
    x = np.concatenate([F_k @ y_k for F_k, y_k, v_k in zip(F, y, visible_objects) if v_k])

    # Add Gaussian noise and constraint image too [-1,1] if necessary
    x = noisy_observations(x, parameters['std_noise']) if parameters['noise_type'] == 'image' else x
    x = constrain_image(x) if parameters['is_constrained'] else x

    #Return both the data and the applied transformation y
    return x, y, F_true

# Compute F^TF for simple computations
def external_product_FTF(F):
    return [np.transpose(F_k, [0, 2, 1]) @ F_k for F_k in F]
# Compute F^Tx for simple computations
def external_product_FTx(x, F):
    return [[x_m @ F_k for x_m in x] for F_k in F]

#Main function that creates the image and all its properties
def create_image(objects, visible_objects, parameters):

    # #Create templates from object description
    templates = list(map(create_template, objects))
    templates = list(map(normalize_template, templates))

    #Generate an image instance
    x, y, F = image_generation(templates, visible_objects, parameters)

    #Get properties of the image
    K = len(objects)
    N_k = list(map(len, templates))
    N = sum(N_k)
    M = np.shape(x)[0]

    #Compute auxiliary external product matrices (for ease of computations)
    FF = external_product_FTF(F)
    F_xm = external_product_FTx(x, F)

    #Return the data object
    return dict({'X_m': x, 'M': M, 'F': F, 'FF': FF, 'F_xm': F_xm,
                  'K': K, 'N_k': N_k, 'N': N, 'y': y, 'objects': objects,
                 'visible_objects': visible_objects,'loaded':False})

def load_image(loaded_data, objects, image_num):

    # Create templates from object description
    templates = list(map(create_template, objects))
    templates = list(map(normalize_template, templates))
    F = list(map(expand_template, templates))

    #Parameter setting
    K = len(objects)
    N_k = list(map(len, templates))
    N = sum(N_k)

    #Transform presence vector into visible objects vector
    gt_presence = loaded_data['gt_presence'][image_num]
    index_k = np.concatenate([np.zeros(1), np.cumsum(N_k)])
    index_k = list(map(int, index_k))
    visible_objects = np.zeros(K, dtype=int)
    for kk in range(K):
        visible_objects[kk] = 1 if gt_presence[index_k[kk]:index_k[kk + 1]].all() else 0

    #Check if the image is filled or not
    if not sum(visible_objects):
        return dict({'visible_objects': visible_objects, 'loaded':True})

    #Create X and y
    gt_points = loaded_data['ground_truth'][image_num]
    x = gt_points[gt_presence==1]
    y = np.zeros([sum(visible_objects), 4]) #We don't have the true transformation
    M = np.shape(x)[0]

    # Compute auxiliary external product matrices (for ease of computations)
    FF = external_product_FTF(F)
    F_xm = external_product_FTx(x, F)

    return dict({'X_m': x, 'M': M, 'F': F, 'FF': FF, 'F_xm': F_xm,
                 'K': K, 'N_k': N_k, 'N': N, 'y': y, 'objects': objects,
                 'visible_objects': visible_objects,'loaded':True})

def figure_data(data_model):

    # Create complete objects for representation
    visible_objects = data_model['visible_objects']

    K, N_k = data_model['K'], data_model['N_k']
    X_m = data_model['X_m']
    index_k = np.concatenate([np.zeros(1), np.cumsum(N_k*visible_objects)])
    index_k = list(map(int, index_k))

    X_figures, F_figures = [], []
    for kk in range(K):
        if visible_objects[kk]:
            if data_model['objects'][kk] != 'L_shape':
                #Concatenate last point to the object to close the shape
                closed_shape = np.concatenate([X_m[index_k[kk]:index_k[kk+1]],
                                               X_m[index_k[kk]].reshape(1,-1)])
            else:
                closed_shape = X_m[index_k[kk]:index_k[kk+1]]
            X_figures.append(closed_shape)
        F_figures.append(data_model['F'][kk] + [data_model['F'][kk][0]])

    return X_figures, F_figures