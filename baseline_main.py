import code
import numpy as np
import utils
import data_creator
import ransac_model as ransac_model
import os
import pickle


def main(main_folder):

    verbose = True
    is_constrained = True

    # Define objects for the images (square, triangle, trapezoid, L_shape, pentagon), {list of str}
    objects = ['square', 'triangle', 'square']
    # objects = ['plane'] + 2*['square']

    # Determine which objects are present in the image, avoiding blank images, {list of {0,1} elements}
    visible_objects = np.random.randint(0, 2, len(objects))
    while not sum(visible_objects):
        visible_objects = np.random.randint(0, 2, len(objects))

    # Generate image and return all data parameters
    parameters = {'is_constrained': True, 'noise_type': 'template', 'std_noise': 0.1}
    data_model = data_creator.create_image(objects, visible_objects, parameters)
    if verbose:
        print('Figure with ' + str(sum(visible_objects)) + ' objects')

    ## Here starts the model

    #Run baseline
    X_obj_est = ransac_model.run(data_model)

    # code.interact(local=dict(globals(), **locals()))

    ## Here starts the plotting of the results

    # Plot the evolution of the VBEM
    figures_dir = main_folder + '/baseline/'
    utils.save_figures_baseline(figures_dir, data_model, X_obj_est, parameters['is_constrained'])

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