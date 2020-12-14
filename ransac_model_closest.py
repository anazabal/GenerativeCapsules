import code
import numpy as np

#Get all sublist of points that form an object, including permutations of the same sublist
#We don't use sets since the order of the elements matter to form the object
def get_possible_objects(data_model, x):

    #Check which objects are available in the experiment
    unique_objects = np.unique(data_model['objects'])
    #Get templates of each unique object
    F_full = []
    for obj in unique_objects:
        index = np.where(np.array(data_model['objects'])==obj)[0][0]
        F_full.append(np.concatenate(data_model['F'][index]))

    # With 2 points we can define a full objects, F becomes a square 4x4 matrix
    F_parts = [F[:4, :] for F in F_full]
    F_inverse = [np.linalg.inv(F) for F in F_parts]

    min_sq = 1e-2
    n_points = len(x)
    points_assigned = []
    X_obj_est_full = []
    cost_total = []
    for ii in range(n_points):

        for jj in range(n_points):

            # If it's the same point, get the next one
            if jj == ii:
                continue

            # Get pair of points to consider, vectorized
            x_vec = x[[ii, jj]].reshape(-1, 1)
            # Compute transformation vector from two points
            y_est = [F @ x_vec for F in F_inverse]
            # Return possible objects given the 2 points used
            x_est = [(F @ y).reshape(-1,2) for F,y in zip(F_full,y_est)]

            # Check conditions for predicted points
            for shape in x_est:
                K = np.shape(shape)[0]
                if K <= 2:
                    raise('Minimum size for objects has to be 3')
                #Compute square differences between the estimated points and the data points
                sq_diffs = np.zeros([n_points,K-2])
                for kk in range(2,K):
                    sq_diffs[:,kk-2] = np.sum((x - shape[kk]) ** 2, 1)

                # #Check if there are points matching the estimated shape
                # cond = (sq_diffs < min_sq)
                # ind_rows, ind_cols = np.where(cond)

                #Get closest points always
                ind_rows = np.argmin(sq_diffs,0)

                # code.interact(local=dict(globals(), **locals()))

                #Avoid detecting itself
                if ii in ind_rows or jj in ind_rows:
                    continue

                #Check that we don't assign the same point twice
                if len(ind_rows) == len(np.unique(ind_rows)):

                    ind_cols = np.arange(0,K-2)
                    cost = np.mean(sq_diffs[ind_rows,ind_cols])

                    #If we don't get as many points as the ones needed to complete the shape, skip
                    # if len(ind_rows) == K-2:
                    # #Ensure that the points detected were not used to create the base
                    # repeated_inds = np.prod([ind in [ii,jj] for ind in ind_rows])
                    # if not repeated_inds:
                    ind_rows_sorted = ind_rows[ind_cols.argsort()]
                    inds_shape = [ii, jj, *ind_rows_sorted.tolist()]
                    # if inds_shape not in points_assigned:
                    points_assigned += [inds_shape]
                    X_obj_est_full += [x[inds_shape]]
                    cost_total += [cost]

    # code.interact(local=dict(globals(), **locals()))

    return points_assigned, X_obj_est_full, cost_total

#Remove redundant patterns composed by the same elements
def find_unique_pattern(x, points_assigned, cost_total):

    #Sort points by their total cost
    index = np.argsort(cost_total)
    points_sorted = np.array(points_assigned)[index]

    close_points_assigned = []
    possible_points = set(np.arange(x.shape[0]))
    found_points = set()

    #Get position of the first time each point is assigned
    first_term = []
    for pp in np.arange(x.shape[0]):
        done = False
        for ind, elem in enumerate(points_sorted):
            if pp in elem:
                first_term += [ind]
                done = True

            if done:
                break

    unique_points_sorted = points_sorted[np.unique(first_term)[::-1]]

    # for elem in unique_points_sorted:
    #     close_points_assigned += [elem]
    #     found_points = set.union(set(elem),found_points)
    #     if found_points == possible_points:
    #         break

    final_points_assigned = []
    for elem in unique_points_sorted:
        if set.issubset(set(elem),possible_points):
            final_points_assigned += [elem]
            possible_points -= set(elem)

        if not possible_points:
            break

    # a = close_points_assigned.copy()
    # # Check unique points in assignments
    # n_points = len(x)
    # final_points_assigned = []
    # while close_points_assigned:
    #     counter = 0
    #     for ii in range(n_points):
    #         find_final_objs = [ii in elem for elem in final_points_assigned]
    #         if sum(find_final_objs) == 1:
    #             # Point already assigned in a unique object
    #             continue
    #         find_objs = [elem for elem in close_points_assigned if ii in elem]
    #         #Check size of objects
    #         K = np.unique([len(elem) for elem in find_objs])
    #         if len(K) != 1:
    #             continue
    #         #Check unique points
    #         base_elems = find_objs[0]
    #         for elem in find_objs:
    #             base_elems = np.intersect1d(base_elems, elem)
    #         if len(base_elems) == K:
    #             # This is a unique object
    #             final_points_assigned += [base_elems]
    #             # Remove elements that contain assigned objects
    #             for obj in find_objs:
    #                 close_points_assigned.remove(obj)
    #             counter += 1
    #
    #     if counter == 0:
    #         break

    # code.interact(local=dict(globals(), **locals()))

    return [x[elem] for elem in final_points_assigned]