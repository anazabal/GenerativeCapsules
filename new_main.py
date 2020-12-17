import code
import numpy as np
import utils
import data_creator
import ransac_model
import os
import pickle
import csv

#TODO - close
def random_initialization(data_model):
    r_mnk = np.random.rand(data_model['M'], data_model['N'])
    return r_mnk / np.sum(r_mnk, 1).reshape(-1, 1)

def uniform_initialization(data_model):
    return np.ones([data_model['M'],data_model['N']]) / data_model['N']

def close_initialization(data_model, factor=0.1):
    M, N, K, N_k = data_model['M'], data_model['N'], data_model['K'], data_model['N_k']
    visible_objects = data_model['visible_objects']

    r_mnk = np.zeros([M, N])
    index_col = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
    index_row = np.concatenate([np.zeros(1), np.cumsum(N_k * visible_objects)]).astype(int)
    # code.interact(local=dict(globals(), **locals()))
    for kk in range(K):
        if visible_objects[kk]:
            # code.interact(local=dict(globals(), **locals()))
            r_mnk[index_row[kk]:index_row[kk + 1], index_col[kk]:index_col[kk + 1]] = np.eye(N_k[kk])

    r_mnk += factor * np.random.rand(M, N)

    return r_mnk / np.sum(r_mnk, 1).reshape(-1, 1)

def r_mnk_initialization(data_model, choice):
    #Possible choices: random, uniform, close
    if choice is 'random':
        return random_initialization(data_model)
    elif choice is 'uniform':
        return uniform_initialization(data_model)
    elif choice is 'close':
        return close_initialization(data_model)
    else:
        raise ValueError('Initialization "' + choice + '" not implemented')


def main(main_folder):

    verbose = True
    video = False

    #Create base objects and their presence or not in the image
    objects = ['square', 'triangle', 'square']
    # visible_objects = np.array([0, 1, 0])

    # objects = 5*['square'] + 5*['triangle']
    visible_objects = np.random.randint(0, 2, len(objects))
    while not sum(visible_objects):
        visible_objects = np.random.randint(0, 2, len(objects))

    # Create image and save all data parameters
    data_parameters = {'is_constrained': True, 'noise_type': 'template', 'std_noise': 0}
    data_model = data_creator.create_image(objects, visible_objects, data_parameters)

    ####################
    # Run baseline and save results
    X_obj_est = ransac_model.run(data_model)

    # code.interact(local=dict(globals(), **locals()))

    ## Here starts the plotting of the results

    # Plot the evolution of the VBEM
    figures_dir = main_folder + '/baseline/'
    utils.save_figures_baseline(figures_dir, data_model, X_obj_est, data_parameters['is_constrained'])

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Save data model
    data_file = figures_dir + 'data_model.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(data_model, f, pickle.HIGHEST_PROTOCOL)
    ######################



    # # Repeat experiment with different initializations of lambda
    # models = 2*['basic_model'] + 2*['basic_model_sinkhorn'] + \
    #          2*['lambda_ML_model'] + 2*['sinkhorn_model_annealing_stop'] + 2*['sinkhorn_model_multiple_restart']
    # lambda_0 = 3*[1,50] + [1,50] + [1,50]

    models = ['sinkhorn_model_annealing_stop']
    lambda_0 = [100]

    inits = 3
    title_inits = ['r_mnk close to truth'] + 20 * ['r_mkn random']
    # r_mnk_method = ['close'] + 19*['random']
    r_mnk_method = ['close'] + 20 * ['random']

    #Get all test multiple initializations for all methods before start
    restarts = 5
    r_mnk = {key: [] for key in range(inits)}
    for ii in range(inits):
        for rr in range(restarts):
            # r_mnk initialization (random, uniform)
            r_mnk[ii].append(r_mnk_initialization(data_model, r_mnk_method[ii]))

    # code.interact(local=dict(globals(), **locals()))

    for ii in range(inits):

        #Compare different models under same initialization
        for mm, model in enumerate(models):

            # Import variational inference model to use
            VI_model = __import__(model)

            #Test different restarts
            final_ELBO = -np.inf
            model_restarts = 1 if model in 'sinkhorn_model_multiple_restart' else restarts

            full_results = []
            for rr in range(model_restarts):

                # Model hyperparameters
                hyper_params = VI_model.hyperparams_initialization(data_model, lambda_0[mm])

                #Save all variables in a dict of lists
                params = VI_model.params_initialization(r_mnk[ii][rr], hyper_params)
                if rr == 0:
                    full_params = dict.fromkeys(params, [])

                # VBEM
                sims = 500
                new_ELBO = -np.inf
                ELBO_epoch = []
                results = []

                for ss in range(sims):

                    #Update parameters and get ELBO
                    if ss == 0:
                        params, ELBO_terms, score = VI_model.params_update(data_model, params, hyper_params, False)
                    else:
                        params, ELBO_terms, score = VI_model.params_update(data_model, params, hyper_params)
                    ELBO = np.sum(ELBO_terms)

                    # Save results
                    results.append([ELBO] + ELBO_terms + [score])

                    if verbose:
                        print("Init {} - Iter {} - ELBO: {:4f}, log_x: {:4f}, KL_Y: {:4f}, KL_Z: {:4f}, KL_pi: {:4f}, Score: {:4f}"
                              .format(ii,ss,ELBO,ELBO_terms[0],-ELBO_terms[1],-ELBO_terms[2],-ELBO_terms[3],score))

                    #stopping criteria
                    if VI_model.stop(ELBO, new_ELBO, hyper_params, params, data_model):
                        break
                    else:
                        new_ELBO = ELBO
                        ELBO_epoch += [ELBO]

                #Check whether the current restart is better than the previous one
                if ELBO > final_ELBO:
                    final_ELBO = ELBO
                    final_params = params
                    final_results = results

                #Save restart
                full_results += results
                for key in params.keys():
                    full_params[key] = full_params[key] + params[key]

                #Exit if solution obtained
                if utils.is_correct(params, data_model):
                    break

            print('Figure ' + str(ii) + ' ' + model  + ' - Restarts: ' + str(rr) + ', Lambda ' + str(lambda_0[mm]) + ': ' + str(utils.is_correct(final_params, data_model)))

            # code.interact(local=dict(globals(), **locals()))

            #Saving results
            figures_dir = main_folder + '/Figures' + str(ii) + '/' + model + '_' + str(lambda_0[mm]) + '/'
            utils.save_results(final_results, data_model, final_params, hyper_params, figures_dir)

            #Plot the evolution of the VBEM
            utils.save_figures(figures_dir, final_results, final_params, data_model)
            #All restarts
            # utils.video_creation(figures_dir, np.array(full_results), full_params, data_model, title_inits[ii], 1000)

            #Only the best
            if video:
                utils.video_creation(figures_dir, np.array(final_results), final_params, data_model, title_inits[ii], 100)

            # code.interact(local=dict(globals(), **locals()))

if __name__ == '__main__':

    #Run this several times
    for pp in range(10):
        main_folder = 'Results/Image_' + str(pp)
        main(main_folder)