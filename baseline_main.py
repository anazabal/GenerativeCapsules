import code
import numpy as np
import utils
import data_creator
import ransac_model as ransac_model
import os
import pickle


def main(main_folder):

    verbose = True

    #Create base objects (square, triangle, trapezoid, L_shape, pentagon), {list of str}
    objects = ['square', 'triangle', 'square']
    objects = 2*['square'] + 2*['triangle'] + 3*['pentagon']

    #Determine which objects are present in the image, avoiding blank images, {list of {0,1} elements}
    visible_objects = np.random.randint(0, 2, len(objects))
    while not sum(visible_objects):
        visible_objects = np.random.randint(0, 2, len(objects))
    # visible_objects = np.array([1,1,1])

    # Create image and save all data parameters
    data_model = data_creator.create_image(objects, visible_objects, is_constrained=False, std_noise=0.1)
    if verbose:
        print('Figure with ' + str(sum(visible_objects)) + ' objects')

    #Randomize image points (Optional)
    x = data_model['X_m']
    # index = np.random.permutation(len(x))
    # x = x[index]

    ## Here starts the model

    #Run baseline
    list_points, list_x, list_costs = ransac_model.get_possible_objects(data_model, x)

    X_obj_est = ransac_model.find_possible_solutions(x, list_points, list_costs)

    # # Check unique points in assignments
    # X_obj_est = ransac_model.find_unique_pattern(x, list_points, list_costs)

    # code.interact(local=dict(globals(), **locals()))

    # Plot the evolution of the VBEM
    figures_dir = main_folder + '/baseline/'
    utils.save_figures_baseline(figures_dir, data_model, X_obj_est, is_constrained=False)

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Save data model
    data_file = figures_dir + 'data_model.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(data_model, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    #Run this several times
    for pp in range(20):
        print('Image ' + str(pp))
        main_folder = 'Results/Image_' + str(pp)
        main(main_folder)