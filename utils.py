import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib._color_data as mcd

import code
import data_creator

def save_figures(figures_dir, data_model, params, is_constrained=True):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Create complete objects for figure plots
    X_figures, F_figures = data_creator.figure_data(data_model)

    # code.interact(local=dict(globals(), **locals()))

    N_k, K, M = data_model['N_k'], data_model['K'], data_model['M']
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(1.*kk/K) for kk in range(K)]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Observed data
    ax[0, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    ax[0, 0].set_title('Input data to the models')

    # Ground truth shapes
    for kk, x_obj in enumerate(X_figures):
        ax[0, 1].plot(x_obj[:, 0], x_obj[:, 1], '-*', color=colors[kk])
    ax[0, 1].set_title('Ground truth assignments')

    # Object assigned to each point (by color)
    points = np.argmax(params['r_mnk'][-1], 1)
    limits = np.concatenate([np.zeros(1), np.cumsum(N_k[:-1])])
    for kk in range(K):
        index = points >= limits[kk]
        ax[1, 0].plot(data_model['X_m'][index, 0], data_model['X_m'][index, 1], '*', color=colors[kk])
        # plot variance (1/lambda) around points
        for center in data_model['X_m'][index]:
            if isinstance(params['lambda_0'][-1], (int, float)):
                std = np.sqrt(1 / params['lambda_0'][-1])
            else:
                std = np.sqrt(1 / params['lambda_0'][-1][kk])
            circle = plt.Circle(center, std, color=colors[kk], alpha=0.1)
            ax[1, 0].add_patch(circle)

    # Object matching
    X_est = [F_figures[kk] @ mu for kk, mu in enumerate(params['mu_k'][-1])]
    for kk, x_obj_est in enumerate(X_est):
        ax[1, 0].plot(x_obj_est[:, 0], x_obj_est[:, 1], color=colors[kk], marker='o')
    # Overlap ground truth to the image, for reference (black points)
    ax[1, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    ax[1, 0].set_title('Predicted object assignments')

    # Get results assignment matrix r_mnk image
    ax[1, 1].imshow(params['r_mnk'][-1], 'gray', vmin=0, vmax=1)
    ax[1, 1].plot([3.5, 3.5], [-0.5, np.shape(params['r_mnk'][-1])[0] - 0.5],color='w')
    ax[1, 1].plot([6.5, 6.5], [-0.5, np.shape(params['r_mnk'][-1])[0] - 0.5], color='w')
    ax[1, 1].set_title('Assignment matrix R')
    ax[1, 1].set_xlabel('Object - Parts')
    ax[1, 1].set_ylabel('Data points')
    ax[1, 1].set_xticks([1.5,5,8.5])
    ax[1, 1].set_xticklabels(['square', 'triangle', 'square'])

    # Set limits for the figures
    if is_constrained:
        ax[0, 0].set_xlim([-1.1, 1.1])
        ax[0, 0].set_ylim([-1.1, 1.1])
        ax[0, 1].set_xlim([-1.1, 1.1])
        ax[0, 1].set_ylim([-1.1, 1.1])
        ax[1, 0].set_xlim([-1.1, 1.1])
        ax[1, 0].set_ylim([-1.1, 1.1])
    else:
        limit = np.max(np.abs(data_model['X_m']))
        ax[0, 0].set_xlim([-limit-0.1, limit+0.1])
        ax[0, 0].set_ylim([-limit-0.1, limit+0.1])
        ax[0, 1].set_xlim([-limit - 0.1, limit + 0.1])
        ax[0, 1].set_ylim([-limit - 0.1, limit + 0.1])
        ax[1, 0].set_xlim([-limit - 0.1, limit + 0.1])
        ax[1, 0].set_ylim([-limit - 0.1, limit + 0.1])

    # code.interact(local=dict(globals(), **locals()))

    fig.savefig(os.path.join(figures_dir, 'output.png'))  # save the figure to file
    plt.close(fig)

def save_figures_ransac(figures_dir, data_model, X_est, is_constrained=True):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Create complete objects for representation
    X_figures, F_figures = data_creator.figure_data(data_model)

    N_k, K, M = data_model['N_k'], data_model['K'], data_model['M']
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(1.*kk/K) for kk in range(K)]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Aggregate points
    ax[0, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    ax[0, 0].set_title('Input data to the models')

    #Ground truth
    for kk, x_obj in enumerate(X_figures):
        ax[0, 1].plot(x_obj[:, 0], x_obj[:, 1], '-*', color=colors[kk])
    ax[0, 1].set_title('Ground truth assignments')

    # Check outcomes
    # ax[1, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    for kk, elem in enumerate(X_est):
        elem_closed = np.concatenate([elem, elem[0].reshape(1, -1)], 0)
        ax[1, 0].plot(elem_closed[:, 0], elem_closed[:, 1], color=colors[kk], marker='o')
    ax[1, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    ax[1, 0].set_title('Predicted object assignments')

    # Set limits for the figures
    if is_constrained:
        ax[0, 0].set_xlim([-1.1, 1.1])
        ax[0, 0].set_ylim([-1.1, 1.1])
        ax[0, 1].set_xlim([-1.1, 1.1])
        ax[0, 1].set_ylim([-1.1, 1.1])
        ax[1, 0].set_xlim([-1.1, 1.1])
        ax[1, 0].set_ylim([-1.1, 1.1])
    else:
        limit = np.max(np.abs(data_model['X_m']))
        ax[0, 0].set_xlim([-limit - 0.1, limit + 0.1])
        ax[0, 0].set_ylim([-limit - 0.1, limit + 0.1])
        ax[0, 1].set_xlim([-limit - 0.1, limit + 0.1])
        ax[0, 1].set_ylim([-limit - 0.1, limit + 0.1])
        ax[1, 0].set_xlim([-limit - 0.1, limit + 0.1])
        ax[1, 0].set_ylim([-limit - 0.1, limit + 0.1])

    fig.savefig(os.path.join(figures_dir, 'output.png'))  # save the figure to file
    plt.close(fig)


def video_creation(figures_dir, results, params, data_model, frames, is_constrained=True):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Create complete objects for representation
    X_figures, F_figures = data_creator.figure_data(data_model)

    #Create movie object
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    N_k, K = data_model['N_k'], data_model['K']
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(1. * kk / K) for kk in range(K)]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Ground truth
    for kk, x_obj in enumerate(X_figures):
        ax[0, 0].plot(x_obj[:, 0], x_obj[:, 1], '-*', color=colors[kk])
    ax[0, 0].set_title('Ground truth')

    # ax[0, 0].set_xlim([-1.1, 1.1])
    # ax[0, 0].set_ylim([-1.1, 1.1])

    # Set limits for the figures
    if is_constrained:
        ax[0, 0].set_xlim([-1.1, 1.1])
        ax[0, 0].set_ylim([-1.1, 1.1])
    else:
        limit = np.max(np.abs(data_model['X_m']))
        ax[0, 0].set_xlim([-limit - 0.1, limit + 0.1])
        ax[0, 0].set_ylim([-limit - 0.1, limit + 0.1])

    # The ELBO is plotted until the last epoch recorded
    ax[0 ,1].set_xlim([0,len(results)])

    # Select number of frames per video
    n_sims = len(params['mu_k'])
    if n_sims > frames:
        n_frames = 1 + n_sims // frames
    else:
        n_frames = 1

    # code.interact(local=dict(globals(), **locals()))

    limits = np.concatenate([np.zeros(1), np.cumsum(N_k[:-1])])
    with writer.saving(fig, figures_dir + "/VI_evolution.mp4", 100):
        for ss in range(0, n_sims, n_frames):

            # ELBO evolution in the first subplot
            ax[0, 1].plot(results[:ss,0])
            ax[0, 1].set_xlabel('lambda: ' + str(params['lambda_0'][ss]))
            ax[0, 1].set_title('ELBO evolution')

            # Object assigned to each point (by color)
            points = np.argmax(params['r_mnk'][ss], 1)
            for kk in range(K):
                index = points >= limits[kk]
                ax[1, 0].plot(data_model['X_m'][index, 0], data_model['X_m'][index, 1], '*', color=colors[kk])
                # Add variance around points
                for center in data_model['X_m'][index]:
                    if isinstance(params['lambda_0'][ss], (int, float)):
                        std = np.sqrt(1 / params['lambda_0'][ss])
                    else:
                        std = np.sqrt(1 / params['lambda_0'][ss][kk])
                    circle = plt.Circle(center, std, color=colors[kk], alpha=0.1)
                    ax[1, 0].add_patch(circle)

            # Object matching
            X_est = [F_figures[kk] @ mu for kk, mu in enumerate(params['mu_k'][ss])]
            for kk, x_obj_est in enumerate(X_est):
                ax[1, 0].plot(x_obj_est[:, 0], x_obj_est[:, 1], color=colors[kk], marker='o')

            ax[1, 0].set_title('Predicted object assignments - Epoch ' + str(ss))

            # ax[1, 0].set_xlabel('Epoch ' + str(ss))
            # Set limits for the figures
            if is_constrained:
                ax[1, 0].set_xlim([-1.1, 1.1])
                ax[1, 0].set_ylim([-1.1, 1.1])
            else:
                limit = np.max(np.abs(data_model['X_m']))
                ax[1, 0].set_xlim([-limit - 0.1, limit + 0.1])
                ax[1, 0].set_ylim([-limit - 0.1, limit + 0.1])

            # Get r_mnk image
            ax[1, 1].imshow(params['r_mnk'][ss], 'gray',vmin=0,vmax=1)
            ax[1, 1].plot([3.5, 3.5], [-0.5, np.shape(params['r_mnk'][-1])[0] - 0.5], color='w')
            ax[1, 1].plot([6.5, 6.5], [-0.5, np.shape(params['r_mnk'][-1])[0] - 0.5], color='w')
            ax[1, 1].set_title('Assignment matrix R')
            ax[1, 1].set_xlabel('Object - Parts')
            ax[1, 1].set_ylabel('Data points')
            ax[1, 1].set_xticks([1.5, 5, 8.5])
            ax[1, 1].set_xticklabels(['square', 'triangle', 'square'])

            #Update frame
            writer.grab_frame()

            # Empty updatable axis
            ax[0, 1].cla()
            ax[1, 0].cla()
            ax[1, 1].cla()

    plt.close(fig)

# Metric based on determinant of r_mnk to determine correctness
def is_correct(params, data_model):

    visible_objects = data_model['visible_objects']
    K, N_k = data_model['K'], data_model['N_k']
    objects = data_model['objects']

    index_row = np.concatenate([np.zeros(1), np.cumsum(N_k*visible_objects)]).astype(int)
    index_col = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
    for kk in range(K):
        if visible_objects[kk]:

            #Check if objects can be matched to one of the templates
            is_obj_k = [obj == objects[kk] for obj in objects]
            index_start = index_col[:-1][is_obj_k]
            index_end = index_col[1:][is_obj_k]

            is_match = False
            for start, end in zip(index_start, index_end):
                perm_matrix = params['r_mnk'][-1][index_row[kk]:index_row[kk+1], start:end]
                abs_det = np.abs(np.linalg.det(perm_matrix))
                if np.abs(abs_det) >= 1/N_k[kk]:
                    #We've found a correct match for this object
                    is_match = True
                    continue

            #If no match for this object, we failed and we don't need to check the rest of the objects
            if not is_match:
                return False

    return True

def save_results(results, data_model, params, hyper_params, figures_dir):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Save data model
    data_file = figures_dir + 'data_model.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(data_model, f, pickle.HIGHEST_PROTOCOL)

    # Save params
    params_file = figures_dir + 'params.pkl'
    with open(params_file, 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)

    # Save hyperparams
    hyperparams_file = figures_dir + 'hyperparams.pkl'
    with open(hyperparams_file, 'wb') as f:
        pickle.dump(hyper_params, f, pickle.HIGHEST_PROTOCOL)

    # Save ELBO results
    save_file = figures_dir + 'results.csv'
    with open(save_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ELBO', 'loglik x', 'KL Y', 'KL Z', 'KL pi', 'score'])
        writer.writerows(results)

    # Save matching result
    save_file = figures_dir + 'matching.csv'
    with open(save_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([is_correct(params, data_model)])

def load_results(figures_dir):

    # Load data model
    data_file = figures_dir + 'data_model.pkl'
    with open(data_file, 'rb') as f:
        data_model = pickle.load(f)

    # Load ELBO results
    save_file = figures_dir + 'results.csv'
    results = []
    with open(save_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            results.append(row)

    return data_model, results