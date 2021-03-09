import numpy as np
import pickle
import os
import pandas as pd
import code
# code.interact(local=dict(globals(), **locals()))

import metrics
from sklearn.metrics import adjusted_rand_score

def load_data_GCM(model, folder, n_images, n_parts, index_k):

    correct_images = 0
    capsule_id_ours = np.zeros([n_images, n_parts])
    pred_presence_ours = np.zeros([n_images, n_parts])
    for im in range(n_images):

        # Remove empty images from the evaluation
        figures_dir = folder + 'Figures' + str(im) + '/' + model + '/'
        if not os.path.exists(figures_dir):
            continue

        # Load data model
        data_file = figures_dir + 'data_model.pkl'
        with open(data_file, 'rb') as f:
            data_model = pickle.load(f)

        # Load data params
        params_file = figures_dir + 'params.pkl'
        with open(params_file, 'rb') as f:
            params = pickle.load(f)

        # Compute predicted presence from r_mnk
        pred_presence_ours[im] = (params['r_mnk'][-1].sum(0) > 1e-3).astype(int)

        # Compute capsules ids from r_mnk
        match_points = np.argmax(params['r_mnk'][-1], 1)

        # Compute capsule ids
        pp = 0
        for dd in range(n_parts):
            cc = sum(index_k <= dd)
            if data_model['visible_objects'][cc]:
                capsule_id_ours[im, dd] = sum(index_k <= match_points[pp])
                pp += 1

        # Add counter
        correct_images += 1

    return capsule_id_ours, pred_presence_ours, correct_images

def main(report_vals, metrics_df, folder, models, n_images):

    # Get relevant fields from loaded CCAE model and validation data
    gt_presence = report_vals['gt_presence'][:n_images]
    pred_presence = report_vals['pred_presence'][:n_images]
    capsule_id = report_vals['capsule_id'][:n_images]
    pattern_id = report_vals['pattern_id'][:n_images]

    # Basic information of the data (square, triangle, square)
    n_parts = 11
    index_k = np.array([4, 7, 11])

    # Load results from each GCM model
    capsule_id_ours = dict()
    pred_presence_ours = dict()
    for mm, model in enumerate(models):
        capsule_id_ours[model], pred_presence_ours[model], correct_images = load_data_GCM(model, folder, n_images, n_parts, index_k)

    # Compute Segmentation Accuracy using ground truth
    index_k = np.array([0, 4, 7, 11])
    presence = np.ones([n_images, 11])
    pattern_true = ((1 + pattern_id) * gt_presence)
    # CCAE segmentation accuracy
    segm_acc_CCAE_true, _, im_acc_CCAE_true, _ = metrics.eval_segmentation((capsule_id + 1) * gt_presence, pattern_true,
                                                                    index_k, presence)
    # GCM-GMM and GCM-DS
    segm_acc_true = []
    image_acc_true = []
    for model in models:
        segm_results = metrics.eval_segmentation((capsule_id_ours[model] + 1) * gt_presence,
                                  pattern_true, index_k, presence)
        segm_acc_true.append(segm_results[0])
        image_acc_true.append(segm_results[2])

    metrics_df['Segm Acc true'] = [segm_acc_CCAE_true] + segm_acc_true

    # Compute Segmentation Accuracy using predictive presence
    segm_acc_CCAE_pred, _, im_acc_CCAE_pred, _ = metrics.eval_segmentation((capsule_id + 1) * metrics.extend_array(pred_presence),
                                                               pattern_true, index_k, presence)
    # GCM-GMM and GCM-DS
    segm_acc_pred = []
    image_acc_pred = []
    for model in models:
        segm_results = metrics.eval_segmentation((capsule_id_ours[model] + 1) * pred_presence_ours[model],
                                                 pattern_true, index_k, presence)
        segm_acc_pred.append(segm_results[0])
        image_acc_pred.append(segm_results[2])
    metrics_df['Segm Acc pred'] = [segm_acc_CCAE_pred] + segm_acc_pred

    # Compute variation of information with ground truth
    VI_true = np.zeros([n_images, 13])
    index_blank = np.ones(n_images, dtype=bool)
    for ii in range(n_images):
        VI_true[ii, 0] = metrics.variation_information_from_assignments(pattern_id[ii], capsule_id[ii], gt_presence[ii],
                                                                gt_presence[ii])
        if (gt_presence[ii] == 0).all():
            index_blank[ii] = False
        for mm, model in enumerate(models):
            VI_true[ii, mm + 1] = metrics.variation_information_from_assignments(pattern_id[ii], capsule_id_ours[model][ii],
                                                                         gt_presence[ii], gt_presence[ii])

    metrics_df['VI true'] = np.mean(VI_true[index_blank], 0)

    # Compute variation of information with predicted presence
    VI = np.zeros([n_images, 13])
    index_blank = np.ones(n_images, dtype=bool)
    pred_presence_extended = metrics.extend_array(pred_presence)
    for ii in range(n_images):
        VI[ii, 0] = metrics.variation_information_from_assignments(pattern_id[ii], capsule_id[ii], gt_presence[ii],
                                                           pred_presence_extended[ii])
        if (gt_presence[ii] == 0).all():
            index_blank[ii] = False
        for mm, model in enumerate(models):
            VI[ii, mm + 1] = metrics.variation_information_from_assignments(pattern_id[ii], capsule_id_ours[model][ii],
                                                                    gt_presence[ii], pred_presence_ours[model][ii])

    metrics_df['VI pred'] = np.mean(VI[index_blank], 0)

    # Compute adjusted rand index with ground truth
    # ARI with gt presence
    pattern_id_missing = np.copy(pattern_id)
    capsule_id_missing = np.copy(capsule_id)
    # capsule_id_ours_missing = np.copy(capsule_id_ours)
    for ii in range(n_images):
        pattern_id_missing[ii][gt_presence[ii] == 0] = -1
        capsule_id_missing[ii][gt_presence[ii] == 0] = -1
        for mm, model in enumerate(models):
            capsule_id_ours[model][ii, gt_presence[ii] == 0] = -1

    # ARI
    adjusted_rand_index_true = np.zeros([n_images, 13])
    index_blank = np.ones(n_images, dtype=bool)
    for ii in range(n_images):
        adjusted_rand_index_true[ii, 0] = adjusted_rand_score(pattern_id_missing[ii], capsule_id_missing[ii])
        if (pattern_id_missing[ii] == -1).all():
            index_blank[ii] = False
        for mm, model in enumerate(models):
            adjusted_rand_index_true[ii, mm + 1] = adjusted_rand_score(pattern_id_missing[ii],
                                                                       capsule_id_ours[model][ii])

    metrics_df['ARI true'] = np.mean(adjusted_rand_index_true[index_blank], 0)

    # Compute adjusted rand index with predicted presence
    pattern_id_missing = np.copy(pattern_id)
    capsule_id_missing = np.copy(capsule_id)
    for ii in range(n_images):
        pattern_id_missing[ii][gt_presence[ii] == 0] = -1
        capsule_id_missing[ii][pred_presence_extended[ii] == 0] = -1
        for mm, model in enumerate(models):
            capsule_id_ours[model][ii, pred_presence_ours[model][ii] == 0] = -1

    # ARI
    adjusted_rand_index = np.zeros([n_images, 13])
    index_blank = np.ones(n_images, dtype=bool)
    for ii in range(n_images):
        adjusted_rand_index[ii, 0] = adjusted_rand_score(pattern_id_missing[ii], capsule_id_missing[ii])
        if (pattern_id_missing[ii] == -1).all():
            index_blank[ii] = False
        for mm, model in enumerate(models):
            adjusted_rand_index[ii, mm + 1] = adjusted_rand_score(pattern_id_missing[ii],
                                                                  capsule_id_ours[model][ii])

    metrics_df['ARI pred'] = np.mean(adjusted_rand_index[index_blank], 0)

    # Compute image accuracy
    image_acc_true = np.array([im_acc_CCAE_true] + image_acc_true) / correct_images
    metrics_df['Image Acc true'] = image_acc_true
    image_acc_pred = np.array([im_acc_CCAE_pred] + image_acc_pred) / correct_images
    metrics_df['Image Acc pred'] = image_acc_pred

    # Save results in a csv
    print(metrics_df)
    metrics_df.to_csv('Test_results.csv')


if __name__ == '__main__':

    # Load data from CCAE file
    with open('valid_report_vals_025.pkl', 'rb') as f:
        report_vals = pickle.load(f)

    # Get folder with saved results and models to evaluate
    folder = 'Test_noise_025/'
    models = ['sinkhorn_model_annealing_stop_50', 'sinkhorn_model_annealing_stop_100',
              'sinkhorn_model_annealing_stop_500', 'sinkhorn_model_annealing_stop_1000',
              'sinkhorn_model_annealing_stop_2000', 'sinkhorn_model_annealing_stop_5000',
              'basic_model_50', 'basic_model_100', 'basic_model_500', 'basic_model_1000', 'basic_model_2000',
              'basic_model_5000']

    # Define output dataframe
    metrics_df = pd.DataFrame(
        index=['CCAE', 'GCM-DS 50', 'GCM-DS 100', 'GCM-DS 500', 'GCM-DS 1000',
               'GCM-DS 2000', 'GCM-DS 5000',
               'GCM-GMM 50', 'GCM-GMM 100', 'GCM-GMM 500', 'GCM-GMM 1000',
               'GCM-GMM 2000', 'GCM-GMM 5000'])

    # Set number of images to evaluate
    n_images = 50

    main(report_vals, metrics_df, folder, models, n_images)