import code
# code.interact(local=dict(globals(), **locals()))
import numpy as np
import data_creator
from itertools import product

def get_bases(data_model):
    # Check which objects are available in the experiment
    unique_objects = np.unique(data_model['objects'])
    # Get templates of each unique object
    templates, F = data_creator.template_generation(unique_objects)
    # With 2 points we can define a full objects, F becomes a square 4x4 matrix
    F_partial = [np.concatenate(F_t[:2],0) for F_t in F]
    bases = [np.linalg.inv(F_p) for F_p in F_partial]

    return bases, F

def get_candidate_objects(x, bases, F, ii, jj):
    # Get pair of points to consider, vectorized
    x_vec = x[[ii, jj]].reshape(-1, 1)
    # Compute transformation vector from two points
    y_est = [B @ x_vec for B in bases]
    # Return possible objects given the 2 points used
    return [(F_t @ y).reshape(-1, 2) for F_t, y in zip(F, y_est)]

def get_possible_permutations(sq_diffs,min_sq):
    #Check if there are points matching the estimated shape
    matching_elements = (sq_diffs <= min_sq)
    #Coorrect if there is at least one match per estimated point
    if (np.sum(matching_elements,0) == 0).any():
        return []
    ind_cols, ind_rows = np.where(matching_elements.T)
    #Group all correct combinations of points
    cols = np.shape(matching_elements)[1]
    groups = [ind_rows[ind_cols==cc] for cc in range(cols)]
    return list(product(*groups))

def square_diffs_2D(A,B):
    rowsA, colsA = np.shape(A)
    rowsB, colsB = np.shape(B)
    if colsA != colsB:
        raise ValueError('Both A and B need to have the same number of columns')
    #Compute sqdiffs between rows in A and B
    sq_diffs = np.zeros([rowsA,rowsB])
    for kk in range(rowsB):
        sq_diffs[:, kk] = np.sum((A - B[kk]) ** 2, 1)

    return sq_diffs


#Get all sublist of points that form an object, including permutations of the same sublist
#We don't use sets since the order of the elements matter to form the object
def get_possible_objects(data_model, x):

    #Get constituting bases and transformation matrices for each object
    bases, F = get_bases(data_model)

    min_sq = 1e-0
    n_points = len(x)
    list_points = []
    list_x = []
    cost_total = []
    for ii in range(n_points):
        for jj in range(n_points):
            # If it's the same point, get the next one
            if jj == ii:
                continue
            #Compute candidate objects for each pair of points
            x_est = get_candidate_objects(x, bases, F, ii, jj)
            # Check each object
            for obj in x_est:
                # Compute square differences between the estimated points and the data points
                sq_diffs = square_diffs_2D(x, obj[2:])
                #Check if there are points matching the estimated shape
                groups = get_possible_permutations(sq_diffs, min_sq)
                #Remove incorrect groups and get the data
                cols = range(np.shape(sq_diffs)[1])
                for group in groups:
                    if ii in group or jj in group:
                        continue
                    elif len(group) != len(np.unique(group)):
                        #Duplicate points in the group
                        continue
                    else:
                        permutation = list((ii,jj) + group)
                        cost_total += [np.mean(sq_diffs[group,cols])]
                        list_points += [permutation]
                        list_x += [x[permutation]]

    return list_points, list_x, cost_total


#Remove duplicate lists of points, which are permutations of each other
def remove_duplicates(list_points, list_costs):
    unique_list_points, unique_sets, unique_costs = [], [], []
    for kk, elem in enumerate(list_points):
        set_elem = set(elem)
        if set_elem not in unique_sets:
            unique_sets += [set_elem]
            unique_list_points += [elem]
            unique_costs += [list_costs[kk]]

    return unique_list_points, unique_costs

def find_possible_solutions(x, list_points, list_costs):
    # Remove duplicate lists of points, which are permutations of each other
    unique_list_points, unique_costs = remove_duplicates(list_points, list_costs)

    #Identify points that appear only once
    aux = unique_list_points
    n_points = np.shape(x)[0]
    solution = []
    costs = []
    not_removed = True
    removed_elements = []
    elements = set(range(n_points))
    while not_removed:
        not_removed = False
        for mm in elements:
            mm_in_sublists = [mm in sublist for sublist in unique_list_points]

            #If there's only 1 option, go to solution
            if sum(mm_in_sublists) == 1:
                pos = mm_in_sublists.index(True)
                pattern = unique_list_points[pos]
                solution += [pattern]
                costs += [unique_costs[pos]]
                removed_elements += pattern
                # Remove element from the list
                del unique_list_points[pos]
                del unique_costs[pos]
                #Remove other sublists with elements already seen in the pattern
                for rr in pattern:
                    rr_in_sublists = [rr in sublist for sublist in unique_list_points]
                    unique_list_points = [unique_list_points[kk] for kk, elem in enumerate(rr_in_sublists) if not elem]
                    unique_costs = [unique_costs[kk] for kk, elem in enumerate(rr_in_sublists) if not elem]
                #We've removed something, we might need to repeat the while
                not_removed = True
            # Remove remaining lists with element mm
            elif mm in removed_elements:
                unique_list_points = [unique_list_points[kk] for kk, elem in enumerate(mm_in_sublists) if not elem]
                unique_costs = [unique_costs[kk] for kk, elem in enumerate(mm_in_sublists) if not elem]

            # print(unique_list_points)
            # #Get this element to the solution
            # elif sum(mm_in_sublists) == 1:
            #     #Get element
            #     pos = mm_in_sublists.index(True)
            #     pattern = unique_list_points[pos]
            #     solution += [pattern]
            #     costs += [unique_costs[pos]]
            #     removed_elements += pattern
            #     #Remove element from the list
            #     del unique_list_points[pos]
            #     del unique_costs[pos]
            #     not_removed = True

        #Update possible elements
        elements = elements - set(removed_elements)

    # code.interact(local=dict(globals(), **locals()))
    #Sort remaining points by cost
    cheap_to_expensive = np.argsort(unique_costs)
    costs_sorted = np.array(unique_costs)[cheap_to_expensive].tolist()
    points_sorted = np.array(unique_list_points)[cheap_to_expensive].tolist()

    #Select a fixed number of possible candidates to get a solution, increase if not achieved
    n_candidates = 10

    if unique_list_points:
        solution_indexes = []
        while not solution_indexes and n_candidates <= 100:
            #Create combinations of solutions given the remaining elements in unique_list_points and solution
            solution_indexes, _ = explore_space_solutions(points_sorted[:n_candidates], [], set(removed_elements), 0, n_points, [])
            n_candidates += 10

        #Collect all solutions and costs, and select the solution with less costs
        if solution_indexes:
            solutions = [solution + np.array(points_sorted)[ind].tolist() for ind in solution_indexes]
            costs = [sum(costs + np.array(costs_sorted)[ind].tolist()) for ind in solution_indexes]
            solution = solutions[np.argmin(costs)]
        else:
            print('Finding solutions failed')

    return [x[elem] for elem in solution]

def explore_space_solutions(list_points, assigned_points, current_set, pos, M, global_solution):
    full_set = set(range(M))
    pos_init = pos
    while pos < len(list_points):
        elem = list_points[pos]
        if current_set.intersection(set(elem)):
            pos += 1
        else:
            possible_set = current_set.union(set(elem))
            if possible_set == full_set:
                # Save solution and remove assigned point
                # global_solution += [assigned_points + [elem]]
                global_solution += [assigned_points + [pos]]
                pos += 1
            else:
                # assigned_points = assigned_points + [elem]
                assigned_points = assigned_points + [pos]
                pos += 1
                global_solution, pos = explore_space_solutions(list_points, assigned_points, possible_set, pos, M, global_solution)
                assigned_points = assigned_points[:-1]

    pos = pos_init
    return global_solution, pos

#Remove redundant patterns composed by the same elements
def find_unique_pattern(x, points_assigned, cost_total):
    # Check unique points in assignments
    n_points = len(x)
    final_points_assigned = []
    while points_assigned:
        counter = 0
        for ii in range(n_points):
            find_final_objs = [ii in elem for elem in final_points_assigned]
            if sum(find_final_objs) == 1:
                # Point already assigned in a unique object
                continue
            find_objs = [elem for elem in points_assigned if ii in elem]
            #Check size of objects
            K = np.unique([len(elem) for elem in find_objs])
            if len(K) != 1:
                continue
            #Check unique points
            base_elems = find_objs[0]
            for elem in find_objs:
                base_elems = np.intersect1d(base_elems, elem)
            if len(base_elems) == K:
                # This is a unique object
                final_points_assigned += [base_elems]
                # Remove elements that contain assigned objects
                for obj in find_objs:
                    points_assigned.remove(obj)
                counter += 1

        if counter == 0:
            break

    return [x[elem] for elem in final_points_assigned]