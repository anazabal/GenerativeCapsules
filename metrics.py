import numpy as np

from scipy.optimize import linear_sum_assignment
from monty.collections import AttrDict

########################## VARIATION OF INFORMATION ########################

def variation_information(X, Y):
    n = sum([len(x) for x in X])
    if n != sum([len(y) for y in Y]):
        raise ValueError('Elements in X and Y do not match')

    VI = 0
    for ii, x in enumerate(X):
        if len(x) == 0:
            continue
        for jj, y in enumerate(Y):
            if len(y) == 0:
                continue
            X_int_Y = np.intersect1d(x, y)
            if len(X_int_Y) == 0:
                continue
            r_ij_p_i = len(X_int_Y) / len(x)
            r_ij_q_j = len(X_int_Y) / len(y)
            r_pq = r_ij_p_i * r_ij_q_j
            VI -= len(X_int_Y) / n * np.log(r_pq)

    return VI


def variation_information_from_assignments(x, y, x_presence, y_presence, n_points=11):
    # Copy arrays!!
    x_c = np.copy(x)
    y_c = np.copy(y)

    # Missing entries need to be encoded!
    x_c[x_presence == 0] = -1
    y_c[y_presence == 0] = -1

    index = np.arange(n_points)
    x_unique = np.unique(x_c, return_inverse=True, return_index=True)
    y_unique = np.unique(y_c, return_inverse=True, return_index=True)

    X = [index[x_unique[2] == kk] for kk in range(len(x_unique[0]))]
    Y = [index[y_unique[2] == kk] for kk in range(len(y_unique[0]))]

    return variation_information(X, Y)

############################ SEGMENTATION ACCURACY ##########################

def extend_array(x):
    out = []
    for row in x:
        out += [4*[row[0]] + 3*[row[1]] + 4*[row[2]]]
    return np.array(out)

def bipartite_match(pred, gt, n_classes=None, presence=None):
  """Does maximum biprartite matching between `pred` and `gt`."""

  if n_classes is not None:
    n_gt_labels, n_pred_labels = np.arange(n_classes), np.arange(n_classes)
  else:
    n_gt_labels = np.unique(gt)
    n_pred_labels = np.unique(pred)

  cost_matrix = np.zeros([max(n_gt_labels)+1, max(n_pred_labels)+1], dtype=np.int32)
  for label in n_gt_labels:
    label_idx = (gt == label)
    for new_label in n_pred_labels:
      errors = np.equal(pred[label_idx], new_label).astype(np.float32)
      if presence is not None:
        errors *= presence[label_idx]

      num_errors = errors.sum()
      cost_matrix[label, new_label] = -num_errors

#   print(cost_matrix)
  row_idx, col_idx = linear_sum_assignment(cost_matrix)
  num_correct_shape = -cost_matrix[row_idx, col_idx]
  num_correct = num_correct_shape.sum()
  acc = float(num_correct) / gt.shape[0]
#   acc = float(num_correct) / (gt!=0).sum()
#   print(num_correct_shape)
#   print(acc)
  return AttrDict(assignment=(row_idx, col_idx), acc=acc,
                  num_correct=num_correct, num_correct_shape=num_correct_shape)


def eval_segmentation(pred, gt, index_k, presence=None):
    """Evaluates segmentation accuracy."""

    if presence is None:
        presence = np.ones_like(gt)

    num_correct = 0
    im_correct = 0
    acc_mean = 0
    shape_correct = np.zeros([pred.shape[0], len(index_k) - 1])
    complete_images = 0
    acc_results = []
    for i in range(pred.shape[0]):
        # Skip empty images
        if (gt[i] == 0).all():
            continue
        complete_images += 1
        res = bipartite_match(pred[i], gt[i], n_classes=4, presence=presence[i])
        # Get accuracy
        acc_mean += res.acc
        acc_results.append(res.acc)
        # Check if full image is correct
        num_correct += res.num_correct
        if res.num_correct == np.sum(presence[i]):
            im_correct += 1
        # Check if partial shapes are correct
        for kk in range(len(index_k) - 1):
            shape_vertices = np.sum(presence[i][index_k[kk]:index_k[kk + 1]])
            if shape_vertices and res.num_correct_shape[kk] == shape_vertices:
                shape_correct[i, kk] = 1

        # Change codes for exchangeability
        if (shape_correct[i] == [1, 0, 0]).all():
            shape_correct[i] = [0, 0, 1]
        if (shape_correct[i] == [1, 1, 0]).all():
            shape_correct[i] = [0, 1, 1]

    #   print(complete_images)
    #   return acc_mean / complete_images, np.float32(float(num_correct) / presence.sum()), im_correct, shape_correct
    return acc_mean / complete_images, acc_results, im_correct, shape_correct

