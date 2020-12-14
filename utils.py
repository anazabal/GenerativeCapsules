import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib._color_data as mcd

import code
import data_creator

def save_figures(figures_dir, results, params, data_model):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Create complete objects for representation
    X_figures, F_figures = data_creator.figure_data(data_model)

    # code.interact(local=dict(globals(), **locals()))

    # colors = ['b', 'r', 'g', 'y', 'm']
    N_k, K, M = data_model['N_k'], data_model['K'], data_model['M']
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(1.*kk/K) for kk in range(K)]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Aggregate points
    ax[0, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    ax[0, 0].set_xlim([-1.1, 1.1])
    ax[0, 0].set_ylim([-1.1, 1.1])

    #Ground truth
    for kk, x_obj in enumerate(X_figures):
        ax[0, 1].plot(x_obj[:, 0], x_obj[:, 1], '-*', color=colors[kk])

    ax[0, 1].set_xlim([-1.1, 1.1])
    ax[0, 1].set_ylim([-1.1, 1.1])

    #Point Assignment
    points = np.argmax(params['r_mnk'][-1], 1)
    limits = np.concatenate([np.zeros(1), np.cumsum(N_k[:-1])])
    # code.interact(local=dict(globals(), **locals()))
    for kk in range(K):
        index = points >= limits[kk]
        ax[1, 0].plot(data_model['X_m'][index, 0], data_model['X_m'][index, 1], '*', color=colors[kk])
        # Add variance around points
        for center in data_model['X_m'][index]:
            if isinstance(params['lambda_0'][-1], (int, float)):
                std = np.sqrt(1 / params['lambda_0'][-1])
            else:
                std = np.sqrt(1 / params['lambda_0'][-1][kk])
            circle = plt.Circle(center, std, color=colors[kk], alpha=0.1)
            ax[1, 0].add_patch(circle)

    #Object matching
    X_est = [F_figures[kk] @ mu for kk, mu in enumerate(params['mu_k'][-1])]
    for kk, x_obj_est in enumerate(X_est):
        ax[1, 0].plot(x_obj_est[:, 0], x_obj_est[:, 1], color=colors[kk], marker='o')
    #Put black points
    ax[1, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')

    ax[1, 0].set_xlabel('Epochs ' + str(len(results)))
    ax[1, 1].set_xlabel('ELBO ' + str(results[-1][0]))
    # ax[1, 0].set_xlim(ax[0, 0].get_xlim())
    # ax[1, 0].set_ylim(ax[0, 0].get_ylim())
    ax[1, 0].set_xlim([-1.1,1.1])
    ax[1, 0].set_ylim([-1.1,1.1])

    # Get r_mnk image
    ax[1, 1].imshow(params['r_mnk'][-1], 'gray', vmin=0, vmax=1)

    fig.savefig(os.path.join(figures_dir, 'output.png'))  # save the figure to file
    plt.close(fig)

def save_figures_baseline(figures_dir, data_model, X_est, is_constrained=True):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Create complete objects for representation
    X_figures, F_figures = data_creator.figure_data(data_model)

    # colors = ['b', 'r', 'g', 'y', 'm']
    N_k, K, M = data_model['N_k'], data_model['K'], data_model['M']
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(1.*kk/K) for kk in range(K)]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Aggregate points
    ax[0, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    if is_constrained:
        ax[0, 0].set_xlim([-1.1, 1.1])
        ax[0, 0].set_ylim([-1.1, 1.1])
    else:
        limit = np.max(np.abs(data_model['X_m']))
        ax[0, 0].set_xlim([-limit-0.1, limit+0.1])
        ax[0, 0].set_ylim([-limit-0.1, limit+0.1])

    #Ground truth
    for kk, x_obj in enumerate(X_figures):
        ax[0, 1].plot(x_obj[:, 0], x_obj[:, 1], '-*', color=colors[kk])
    if is_constrained:
        ax[0, 1].set_xlim([-1.1, 1.1])
        ax[0, 1].set_ylim([-1.1, 1.1])
    else:
        # limit = max(data_model['X_m'])
        ax[0, 1].set_xlim([-limit-0.1, limit+0.1])
        ax[0, 1].set_ylim([-limit-0.1, limit+0.1])

    # Check outcomes
    ax[1, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    # index = np.where(data_model['visible_objects']==1)[0]
    # code.interact(local=dict(globals(), **locals()))
    # for ii, elem in enumerate(X_est):
    #     if data_model['objects'][index[ii]] != 'L_shape':
    #         elem_closed = np.concatenate([elem, elem[0].reshape(1, -1)], 0)
    #     else:
    #         elem_closed = elem
    #     ax[1, 0].plot(elem_closed[:, 0], elem_closed[:, 1])
    for ii, elem in enumerate(X_est):
        elem_closed = np.concatenate([elem, elem[0].reshape(1, -1)], 0)
        ax[1, 0].plot(elem_closed[:, 0], elem_closed[:, 1])
    if is_constrained:
        ax[1, 0].set_xlim([-1.1, 1.1])
        ax[1, 0].set_ylim([-1.1, 1.1])
    else:
        # limit = max(data_model['X_m'])
        ax[1, 0].set_xlim([-limit-0.1, limit+0.1])
        ax[1, 0].set_ylim([-limit-0.1, limit+0.1])

    fig.savefig(os.path.join(figures_dir, 'output.png'))  # save the figure to file
    plt.close(fig)


def video_creation(figures_dir, results, params, data_model, title, frames):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Create complete objects for representation
    X_figures, F_figures = data_creator.figure_data(data_model)

    #Create movie object
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    # colors = ['b', 'r', 'g', 'y', 'm']
    N_k, K = data_model['N_k'], data_model['K']
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(1. * kk / K) for kk in range(K)]

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    # Ground truth
    for kk, x_obj in enumerate(X_figures):
        ax[0, 0].plot(x_obj[:, 0], x_obj[:, 1], '-*', color=colors[kk])

    x_limits = ax[0, 0].get_xlim()
    y_limits = ax[0, 0].get_ylim()
    ax[0, 0].set_xlim([-1.1, 1.1])
    ax[0, 0].set_ylim([-1.1, 1.1])

    ax[0 ,1].set_xlim([0,len(results)])
    fig.suptitle(title)

    #Select number of frames per video
    n_sims = len(params['mu_k'])
    if n_sims > frames:
        n_frames = 1 + n_sims // frames
    else:
        n_frames = 1

    limits = np.concatenate([np.zeros(1), np.cumsum(N_k[:-1])])
    with writer.saving(fig, figures_dir + "/VI_evolution.mp4", 100):
        for ss in range(0, n_sims, n_frames):

            # ELBO evolution in the first subplot
            ax[0, 1].plot(results[:ss,0])
            ax[0, 1].set_xlabel('lambda: ' + str(params['lambda_0'][ss]))

            # Point assignment
            points = np.argmax(params['r_mnk'][ss + 1], 1)
            for kk in range(K):
                index = points >= limits[kk]
                ax[1, 0].plot(data_model['X_m'][index, 0], data_model['X_m'][index, 1], '*', color=colors[kk])
                #Add variance around points
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

            ax[1, 0].set_xlabel('Epoch ' + str(ss))
            # ax[1, 0].set_xlim(x_limits)
            # ax[1, 0].set_ylim(y_limits)
            ax[1, 0].set_xlim([-1.1, 1.1])
            ax[1, 0].set_ylim([-1.1, 1.1])

            # Get r_mnk image
            ax[1, 1].imshow(params['r_mnk'][ss + 1], 'gray',vmin=0,vmax=1)

            # Titles
            ax[0, 0].set_title('Ground truth')
            ax[0, 1].set_title('ELBO evolution')
            ax[1, 0].set_title('VI evolution')
            ax[1, 1].set_title('r_mnk mask')

            #Update frame
            writer.grab_frame()

            # Empty updatable axis
            ax[0, 1].cla()
            ax[1, 0].cla()
            ax[1, 1].cla()

    plt.close(fig)

# #Metric based on determinant of r_mnk to determine correctness
# def is_correct(params, data_model):
#
#     visible_objects = data_model['visible_objects']
#     K, N_k = data_model['K'], data_model['N_k']
#     objects = data_model['objects']
#
#     index_row = np.concatenate([np.zeros(1), np.cumsum(N_k*visible_objects)]).astype(int)
#     index_col = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
#     # code.interact(local=dict(globals(), **locals()))
#     for kk in range(K):
#         code.interact(local=dict(globals(), **locals()))
#         if visible_objects[kk]:
#             #Square
#             if kk == 0 or kk == 2:
#                 M1 = params['r_mnk'][-1][index_row[kk]:index_row[kk+1], :4]
#                 M2 = params['r_mnk'][-1][index_row[kk]:index_row[kk+1], 7:]
#
#                 abs_det_M1 = np.abs(np.linalg.det(M1))
#                 abs_det_M2 = np.abs(np.linalg.det(M2))
#
#                 if np.abs(abs_det_M1) < 1/N_k[kk] and np.abs(abs_det_M2) < 1/N_k[kk]:
#                     return False
#             #Triangle
#             else:
#                 M3 = params['r_mnk'][-1][index_row[kk]:index_row[kk+1], index_col[kk]:index_col[kk+1]]
#                 abs_det_M3 = np.abs(np.linalg.det(M3))
#                 if np.abs(abs_det_M3) < 1/N_k[kk]:
#                     return False
#
#     return True

#Metric based on determinant of r_mnk to determine correctness
def is_correct(params, data_model):

    visible_objects = data_model['visible_objects']
    K, N_k = data_model['K'], data_model['N_k']
    objects = data_model['objects']

    index_row = np.concatenate([np.zeros(1), np.cumsum(N_k*visible_objects)]).astype(int)
    index_col = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
    # code.interact(local=dict(globals(), **locals()))
    for kk in range(K):
        # code.interact(local=dict(globals(), **locals()))
        if visible_objects[kk]:

            #Check if objects can be match to one of the templates
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

            #If no match for this object, we failed and we don't need to check teh rest of the objects
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





def save_figures_edges(figures_dir, results, params, data_model):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Create complete objects for representation
    X_figures, F_figures = data_creator.figure_data(data_model)

    # code.interact(local=dict(globals(), **locals()))

    # colors = ['b', 'r', 'g', 'y', 'm']
    N_k, K, M = data_model['N_k'], data_model['K'], data_model['M']
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(1.*kk/K) for kk in range(K)]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

    # Aggregate points
    ax[0, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')
    ax[0, 0].set_xlim([-1.1, 1.1])
    ax[0, 0].set_ylim([-1.1, 1.1])

    #Ground truth
    for kk, x_obj in enumerate(X_figures):
        ax[0, 1].plot(x_obj[:, 0], x_obj[:, 1], '-*', color=colors[kk])

    ax[0, 1].set_xlim([-1.1, 1.1])
    ax[0, 1].set_ylim([-1.1, 1.1])

    # #Point Assignment
    # points = np.argmax(params['r_mnk'][-1], 1)
    # limits = np.concatenate([np.zeros(1), np.cumsum(N_k[:-1])])
    # code.interact(local=dict(globals(), **locals()))
    # for kk in range(K):
    #     index = points >= limits[kk]
    #     ax[1, 0].plot(data_model['X_m'][index, 0], data_model['X_m'][index, 1], '*', color=colors[kk])
    #     # Add variance around points
    #     for center in data_model['X_m'][index]:
    #         if isinstance(params['lambda_0'][-1], (int, float)):
    #             std = np.sqrt(1 / params['lambda_0'][-1])
    #         else:
    #             std = np.sqrt(1 / params['lambda_0'][-1][kk])
    #         circle = plt.Circle(center, std, color=colors[kk], alpha=0.1)
    #         ax[1, 0].add_patch(circle)

    #Object matching
    X_est = [F_figures[kk] @ mu for kk, mu in enumerate(params['mu_k'][-1])]
    for kk, x_obj_est in enumerate(X_est):
        ax[1, 0].plot(x_obj_est[:, 0], x_obj_est[:, 1], color=colors[kk], marker='o')
    #Put black points
    ax[1, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')

    ax[1, 0].set_xlabel('Epochs ' + str(len(results)))
    ax[1, 1].set_xlabel('ELBO ' + str(results[-1][0]))
    # ax[1, 0].set_xlim(ax[0, 0].get_xlim())
    # ax[1, 0].set_ylim(ax[0, 0].get_ylim())
    ax[1, 0].set_xlim([-1.1,1.1])
    ax[1, 0].set_ylim([-1.1,1.1])

    # Get r_mnk image
    ax[1, 1].imshow(params['r_mnk'][-1], 'gray', vmin=0, vmax=1)

    # # Get r_mnk image
    # ax[1, 2].imshow(params['s_mm'][-1], 'gray', vmin=0, vmax=1)

    fig.savefig(os.path.join(figures_dir, 'output.png'))  # save the figure to file
    plt.close(fig)


def video_creation_edges(figures_dir, results, params, data_model, title, frames):

    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Create complete objects for representation
    X_figures, F_figures = data_creator.figure_data(data_model)

    #Create movie object
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    # colors = ['b', 'r', 'g', 'y', 'm']
    N_k, K = data_model['N_k'], data_model['K']
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(1. * kk / K) for kk in range(K)]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

    # Ground truth
    for kk, x_obj in enumerate(X_figures):
        ax[0, 0].plot(x_obj[:, 0], x_obj[:, 1], '-*', color=colors[kk])

    x_limits = ax[0, 0].get_xlim()
    y_limits = ax[0, 0].get_ylim()
    ax[0, 0].set_xlim([-1.1, 1.1])
    ax[0, 0].set_ylim([-1.1, 1.1])

    ax[0 ,1].set_xlim([0,len(results)])
    fig.suptitle(title)

    #Select number of frames per video
    n_sims = len(params['mu_k'])
    if n_sims > frames:
        n_frames = 1 + n_sims // frames
    else:
        n_frames = 1

    limits = np.concatenate([np.zeros(1), np.cumsum(N_k[:-1])])
    with writer.saving(fig, figures_dir + "/VI_evolution.mp4", 100):
        for ss in range(0, n_sims, n_frames):

            # ELBO evolution in the first subplot
            ax[0, 1].plot(results[:ss,0])
            ax[0, 1].set_xlabel('lambda: ' + str(params['lambda_0'][ss]))

            # # Point assignment
            # points = np.argmax(params['r_mnk'][ss + 1], 1)
            # for kk in range(K):
            #     index = points >= limits[kk]
            #     ax[1, 0].plot(data_model['X_m'][index, 0], data_model['X_m'][index, 1], '*', color=colors[kk])
            #     #Add variance around points
            #     for center in data_model['X_m'][index]:
            #         if isinstance(params['lambda_0'][ss], (int, float)):
            #             std = np.sqrt(1 / params['lambda_0'][ss])
            #         else:
            #             std = np.sqrt(1 / params['lambda_0'][ss][kk])
            #         circle = plt.Circle(center, std, color=colors[kk], alpha=0.1)
            #         ax[1, 0].add_patch(circle)

            # Object matching
            X_est = [F_figures[kk] @ mu for kk, mu in enumerate(params['mu_k'][ss])]
            for kk, x_obj_est in enumerate(X_est):
                ax[1, 0].plot(x_obj_est[:, 0], x_obj_est[:, 1], color=colors[kk], marker='o')
            # Put black points
            ax[1, 0].plot(data_model['X_m'][:, 0], data_model['X_m'][:, 1], 'k*')

            ax[1, 0].set_xlabel('Epoch ' + str(ss))
            # ax[1, 0].set_xlim(x_limits)
            # ax[1, 0].set_ylim(y_limits)
            ax[1, 0].set_xlim([-1.1, 1.1])
            ax[1, 0].set_ylim([-1.1, 1.1])

            # Get r_mnk image
            ax[1, 1].imshow(params['r_mnk'][ss + 1], 'gray',vmin=0,vmax=1)

            # # Get r_mnk image
            # ax[1, 2].imshow(params['s_mm'][ss + 1], 'gray', vmin=0, vmax=1)

            # Titles
            ax[0, 0].set_title('Ground truth')
            ax[0, 1].set_title('ELBO evolution')
            ax[1, 0].set_title('VI evolution')
            ax[1, 1].set_title('r_mnk mask')

            #Update frame
            writer.grab_frame()

            # Empty updatable axis
            ax[0, 1].cla()
            ax[1, 0].cla()
            ax[1, 1].cla()

    plt.close(fig)