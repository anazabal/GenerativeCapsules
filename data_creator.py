import numpy as np
import pickle
import code

#Templates for different objects
def create_square():
    return np.array([[-np.sqrt(2) / 2, np.sqrt(2) / 2], [np.sqrt(2) / 2, np.sqrt(2) / 2],
                       [np.sqrt(2) / 2, -np.sqrt(2) / 2], [-np.sqrt(2) / 2, -np.sqrt(2) / 2]])
def create_triangle():
    return np.array([[-np.sqrt(3) / 2, -0.5], [0, 1], [np.sqrt(3) / 2, -0.5]])
    # return np.array([[1, 2], [3, 1], [3, 3]])
def create_trapezoid():
    return np.array([[-0.5, 1], [0.5, 1], [1, -1], [-1, -1]])
def create_pentagon():
    return np.array([[-0.5, 0], [0, -1], [0.5, -1], [0.5, 0.5], [0, 1]])
def create_L():
    return np.array([[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1]])

def create_object(obj):
    object_creator = {'square':create_square(),'triangle':create_triangle(),'trapezoid':create_trapezoid(),
               'pentagon':create_pentagon(),'L_shape':create_L()}
    if obj in object_creator.keys():
        return object_creator[obj]
    else:
        raise ValueError('Unimplemented object "' + obj + '"')

# This function creates the transformation matrix for a given 2D-point {list of 2x4 numpy arrays, number of vertices}
def transformation_matrix(point):
    return [np.array([[1,0,x,y],[0,1,y,-x]]) for x,y in point]

#Generate the object templates and their corresponding arrays of transformation matrices
def template_generation(objects):
    templates = [create_object(obj) for obj in objects]
    F = [transformation_matrix(template) for template in templates]
    return templates, F

def constrain_image(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1

def image_generation(F, visible_objects, is_constrained=True, std_noise=0):
    # Create transformed data points with random transformations
    y = np.random.randn(len(visible_objects), 4)
    x_m = np.concatenate([F_k @ y_k for F_k, y_k, v_k in zip(F,y,visible_objects) if v_k])

    # Add Gaussian noise
    x_m += std_noise * np.random.randn(x_m.shape[0], 2)
    # Enforce all the points to be in [-1,1]
    x_m = constrain_image(x_m) if is_constrained else x_m

    return x_m, y

#Main function that creates the image and all its properties
def create_image(objects, visible_objects, is_constrained=True, std_noise=0):

    # Create base objects
    templates, F = template_generation(objects)
    # Create image
    x, y = image_generation(F, visible_objects, is_constrained, std_noise)

    # code.interact(local=dict(globals(), **locals()))

    # Get main properties
    K = len(objects)
    N_k = np.array([len(obj) for obj in templates])
    N = sum(N_k)
    M = np.shape(x)[0]

    # Create F^TF and F^Tx_m matrices (simplifies computations later)
    FF = [np.transpose(F_k, [0, 2, 1]) @ F_k for F_k in F]
    F_xm = [[x_m @ F_k for x_m in x] for F_k in F]

    # code.interact(local=dict(globals(), **locals()))

    return dict({'X_m': x, 'M': M, 'F': F, 'FF': FF, 'F_xm': F_xm,
                  'K': K, 'N_k': N_k, 'N': N, 'y': y, 'objects': objects,
                 'visible_objects': visible_objects,'loaded':False})









def load_image(objects, image_num):

    # Load data from SCAE
    with open('report_vals.pkl', 'rb') as f:
        report_vals = pickle.load(f)

    #Load points and presence vector
    gt_points = report_vals['ground_truth'][image_num]
    gt_presence = report_vals['gt_presence'][image_num]

    # Create base objects
    templates, F = template_generation(objects)
    _, dim_y = np.shape(F[0][0])

    #Parameter setting
    K = 3
    N_k = np.array([4,3,4])
    N = np.sum(N_k)

    #Transform presence vector into visible objects vector
    index_k = np.concatenate([np.zeros(1),np.cumsum(N_k)])
    index_k = [int(x) for x in index_k]
    visible_objects = np.array([])
    # code.interact(local=dict(globals(), **locals()))
    for kk in range(K):
        if gt_presence[index_k[kk]:index_k[kk+1]].all():
            visible_objects = np.concatenate([visible_objects,np.ones(1)])
        else:
            visible_objects = np.concatenate([visible_objects, np.zeros(1)])

    visible_objects = np.array([int(x) for x in visible_objects])

    #Check if the image is filled or not
    if not sum(visible_objects):
        return dict({'visible_objects': visible_objects,'loaded':True})

    #Create X and y
    X = []
    for kk in range(K):
        if visible_objects[kk]:
            X.append(gt_points[index_k[kk]:index_k[kk+1]])
    # code.interact(local=dict(globals(), **locals()))
    y = np.zeros([np.sum(visible_objects),4]) #We don't have the true transformation
    X_m = np.concatenate(X)

    # Aggregate points
    M = np.shape(X_m)[0]

    # Create F^TF matrices
    FF = [np.transpose(F[kk], [0, 2, 1]) @ F[kk] for kk in range(K)]

    # Create F^Tx_m products
    F_xm = [[np.transpose(F[kk], [0, 2, 1]) @ X_m[mm] for mm in range(M)] for kk in range(K)]

    return dict({'X_m': X_m, 'M': M, 'F': F, 'FF': FF, 'F_xm': F_xm,
                 'K': K, 'N_k': N_k, 'N': N, 'y': y, 'objects': objects,
                 'visible_objects': visible_objects,'loaded':True})

def figure_data(data_model):

    # Create complete objects for representation
    visible_objects = data_model['visible_objects']

    K, N_k = data_model['K'], data_model['N_k']
    X_m = data_model['X_m']
    index_k = np.concatenate([np.zeros(1), np.cumsum(N_k*visible_objects)])
    index_k = [int(x) for x in index_k]

    X_figures = []
    F_figures = []
    # code.interact(local=dict(globals(), **locals()))
    for kk in range(K):
        if visible_objects[kk]:
            if data_model['objects'][kk] != 'L_shape':
                closed_shape = np.concatenate([X_m[index_k[kk]:index_k[kk+1]], X_m[index_k[kk]].reshape(1,-1)])
            else:
                closed_shape = X_m[index_k[kk]:index_k[kk+1]]
            X_figures.append(closed_shape)
        F_figures.append(data_model['F'][kk] + [data_model['F'][kk][0]])

    return X_figures, F_figures