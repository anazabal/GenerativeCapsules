import code
import numpy as np
import utils
import data_creator
import ransac_model
import os
import pickle
import test
import csv

def main(main_folder, noise, n_images, noise_file=None):

    verbose = False
    plot_figures = True
    video = False

    #Create base objects and their presence or not in the image
    objects = ['square',
               'triangle',
               'square',
               ]
    # Create image and save all data parameters
    data_parameters = {'is_constrained': True,
                       'noise_type': 'template',
                       'std_noise': noise,
                       }
    # Models to test with different initial values of lambda
    models = 6 * ['basic_model'] + 6 * ['sinkhorn_model_annealing_stop']
    model_modules = list(map(__import__, models))
    lambda_0 = 2 * [50, 100, 500, 1000, 2000, 5000]

    #Load CCAE images dataset
    if noise_file is not None:
        with open(noise_file, 'rb') as f:
            loaded_data = pickle.load(f)

    sims = 500
    restarts = 5
    r_mnk = {key: [] for key in range(n_images)}
    for ii in range(n_images):

        # Get image from CCAE dataset or generate one from the generator
        if noise_file is not None:
            data_model = data_creator.load_image(loaded_data, objects, ii)
        else:
            # Generate a random visible objects vector, removing blank images
            visible_objects = np.random.randint(0, 2, len(objects))
            while not sum(visible_objects):
                visible_objects = np.random.randint(0, 2, len(objects))
            # Create image
            data_model = data_creator.create_image(objects, visible_objects, data_parameters)

        # Continue if the image is blank
        if not sum(data_model['visible_objects']):
            print('Figure ' + str(ii) + ' empty, skipping')
            continue

        ####################
        # Run RANSAC and save results
        X_obj_est, assignment, X_transformed = ransac_model.run(data_model)

        # Plot the evolution of the VBEM
        save_folder = main_folder + '/Figures' + str(ii) + '/RANSAC/'
        if plot_figures:
            utils.save_figures_ransac(save_folder, data_model, X_transformed, data_parameters['is_constrained'])

        #Save data model
        ransac_model.save_results(save_folder, data_model, X_obj_est, X_transformed, assignment)
        ######################

        # Get all test multiple initializations for all methods before start.
        # All methods use the same random initialization function
        for rr in range(restarts):
            r_mnk[ii].append(model_modules[0].r_mnk_initialization(data_model))

        # Compute GCM-perm for the same image with a different initialization of r_mnk
        #Compare different models under same initialization
        for mm, model in enumerate(models):

            #Test different restarts
            final_ELBO = -np.inf
            for rr in range(restarts):

                # Set model hyperparameters
                hyper_params = model_modules[mm].hyperparams_initialization(data_model, lambda_0=lambda_0[mm])

                # Initialize dict with all variables necessary for the inference
                params = model_modules[mm].params_initialization(r_mnk[ii][rr], data_model, hyper_params)

                # VBEM
                new_ELBO = -np.inf
                ELBO_epoch = [new_ELBO]
                results = []
                for ss in range(sims):

                    #Update parameters and get ELBO
                    params, ELBO_terms, score = model_modules[mm].params_update(data_model, params, hyper_params)
                    ELBO = np.sum(ELBO_terms)

                    # Save results
                    results.append([ELBO] + ELBO_terms + [score])

                    if verbose and not ss % 50:
                        print("Init {} - Iter {} - ELBO: {:4f}, log_x: {:4f}, KL_Y: {:4f}, KL_Z: {:4f}, Score: {:4f}"
                              .format(ii,ss,ELBO,ELBO_terms[0],-ELBO_terms[1],-ELBO_terms[2],score))

                    #stopping criteria
                    ELBO_epoch += [ELBO]
                    if model_modules[mm].stop(ELBO_epoch, new_ELBO, hyper_params, params, data_model, lambda_0[mm]):
                        break
                    else:
                        new_ELBO = ELBO

                # code.interact(local=dict(globals(), **locals()))

                # Check whether the current restart is better than the previous one
                if ELBO > final_ELBO:
                    final_ELBO = ELBO
                    final_params = params
                    final_results = results

                # Exit if solution obtained (not necessary, but speeds up computation)
                if utils.is_correct(params, data_model):
                    break

            print('Figure ' + str(ii) + ' ' + model  + ', Lambda ' + str(lambda_0[mm]) + \
                  ' - Restarts: ' + str(rr) + ': ' + str(utils.is_correct(final_params, data_model)))

            # code.interact(local=dict(globals(), **locals()))

            #Saving results
            save_folder = main_folder + '/Figures' + str(ii) + '/' + model + '_' + str(lambda_0[mm]) + '/'
            utils.save_results(final_results, data_model, final_params, hyper_params, save_folder)

            #Plot the evolution of the VBEM
            if plot_figures:
                utils.save_figures(save_folder, data_model, final_params, data_parameters['is_constrained'])
            # Create video with the inference evolution
            if video:
                utils.video_creation(save_folder, np.array(final_results), final_params, data_model, frames=1000)

            # code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':

    # Run this several times
    n_images = 50
    # noise_levels = [0.01, 0.05, 0.1, 0.5]
    noise_levels = [0]
    for noise in noise_levels:
        main_folder = 'Test_noise_025'
        main(main_folder, noise, n_images)
        # main(main_folder, noise, n_images, 'valid_report_vals_025.pkl')