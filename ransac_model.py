import code
# code.interact(local=dict(globals(), **locals()))
import numpy as np
import os
import pickle
import data_creator
from itertools import product
from functools import reduce

def get_bases(data_model):
    # Check which objects are available in the experiment
    unique_objects = np.unique(data_model['objects'])
    # Get templates of each unique object
    templates = [data_creator.create_template(obj) for obj in unique_objects]
    F = [data_creator.expand_template(template) for template in templates]
    # With 2 points we can define a full objects, F becomes a square 4x4 matrix
    F_partial = [np.concatenate(F_t[:2],0) for F_t in F]
    bases = [np.linalg.inv(F_p) for F_p in F_partial]

    return bases, F

# Given 2 points vectorized, compute the estimated affine transformation y = B@x_vec,
# and the estimated points of the object x_est = F @ y_est
def get_candidate_objects(x_vec, bases, F):
    y_est = [B @ x_vec for B in bases]
    # Return possible objects given the 2 points used and available templates
    return [(F_t @ y).reshape(-1, 2) for F_t, y in zip(F, y_est)]

# Function that returns all possible point permutations that are consistent with the image
def get_possible_permutations(sq_diffs, min_sq):
    #Check if there are points matching the estimated shape
    matching_elements = (sq_diffs <= min_sq)
    #Only correct if there is at least one match per estimated point (column)
    if (np.sum(matching_elements, 0) == 0).any():
        return []
    ind_cols, ind_rows = np.where(matching_elements.T)
    #Group all correct combinations of points
    cols = np.shape(matching_elements)[1]
    groups = [ind_rows[ind_cols==cc] for cc in range(cols)]
    return list(product(*groups)) #TODO: this can be inefficient if groups is large

def square_diffs_2D(A, B):
    rowsA, colsA = np.shape(A)
    rowsB, colsB = np.shape(B)
    if colsA != colsB:
        raise ValueError('Both A and B need to have the same number of columns')

    sq_diffs = np.zeros([rowsA,rowsB])
    for kk in range(rowsB):
        sq_diffs[:, kk] = np.sum((A - B[kk]) ** 2, 1)

    return sq_diffs


#Get all sublist of points that form an object, including permutations of the same sublist
#We don't use sets since the order of the elements matter to form the object
def get_possible_objects(data_model, tol=1e-0):

    #Get constituting bases and transformation matrices for each object
    bases, F = get_bases(data_model)

    n_points = data_model['M']
    list_points, list_x, cost_total, x_est_total = [], [], [], []
    #Loop twice through all points, order is important
    for ii in range(n_points):
        for jj in range(n_points):
            # If it's the same point, get the next one
            if jj == ii:
                continue
            #Compute candidate objects for each pair of points
            x_2points_vec = data_model['X_m'][[ii, jj]].reshape(-1, 1)
            x_est = get_candidate_objects(x_2points_vec, bases, F)
            # Check each object
            for obj in x_est:
                # Compute square differences between the data points and the estimated points not used to form the base
                sq_diffs = square_diffs_2D(data_model['X_m'], obj[2:])
                # code.interact(local=dict(globals(), **locals()))
                #Check if there are points in the image matching the estimated shape
                groups = get_possible_permutations(sq_diffs, tol)
                # code.interact(local=dict(globals(), **locals()))
                #Remove incorrect groups and get the data
                cols = range(np.shape(sq_diffs)[1])
                for group in groups:
                    if ii in group or jj in group:
                        #Match to points used to create the base
                        continue
                    elif len(group) != len(np.unique(group)):
                        #Duplicate points in the group
                        continue
                    else:
                        permutation = list((ii,jj) + group)
                        cost_total += [np.mean(sq_diffs[group,cols])]
                        list_points += [permutation]
                        list_x += [data_model['X_m'][permutation]]
                        x_est_total += [obj]

    return list_points, list_x, cost_total, x_est_total


#Remove duplicate lists of points, which are permutations of each other
def remove_duplicates(list_points, list_costs, x_est_total):
    unique_list_points, unique_sets, unique_costs, unique_x_est = [], [], [], []
    for kk, elem in enumerate(list_points):
        set_elem = set(elem)
        if set_elem not in unique_sets:
            unique_sets += [set_elem]
            unique_list_points += [elem]
            unique_costs += [list_costs[kk]]
            unique_x_est += [x_est_total[kk]]
        else:
            index_duplicate = unique_sets.index(set_elem)
            #Get the set with lower sq_diff
            if list_costs[kk] < unique_costs[index_duplicate]:
                unique_sets[index_duplicate] = set_elem
                unique_list_points[index_duplicate] = elem
                unique_costs[index_duplicate] = list_costs[kk]
                unique_x_est[index_duplicate] = x_est_total[kk]

    return unique_list_points, unique_costs, unique_x_est

def identify_unique_permutations(x, list_points, list_costs, list_x_est):

    solution = []
    costs = []
    x_est = []
    not_removed = True
    removed_elements = []
    elements = set(range(np.shape(x)[0]))
    new_list_points = list_points.copy()
    new_list_costs = list_costs.copy()
    new_list_x_est = list_x_est.copy()

    while not_removed:
        not_removed = False
        for mm in elements:
            mm_in_sublists = [mm in sublist for sublist in new_list_points]

            # If there's only 1 option, go to solution
            if sum(mm_in_sublists) == 1:
                pos = mm_in_sublists.index(True)
                pattern = new_list_points[pos]
                solution += [pattern]
                costs += [new_list_costs[pos]]
                x_est += [new_list_x_est[pos]]
                removed_elements += pattern
                # Remove element from the list
                del new_list_points[pos]
                del new_list_costs[pos]
                del new_list_x_est[pos]
                # Remove other sublists with elements already seen in the pattern
                for rr in pattern:
                    rr_in_sublists = [rr in sublist for sublist in new_list_points]
                    new_list_points = [new_list_points[kk] for kk, elem in enumerate(rr_in_sublists) if not elem]
                    new_list_costs = [new_list_costs[kk] for kk, elem in enumerate(rr_in_sublists) if not elem]
                    new_list_x_est = [new_list_x_est[kk] for kk, elem in enumerate(rr_in_sublists) if not elem]
                # We've removed something, we might need to repeat the while
                not_removed = True
            # Remove remaining lists with element mm
            elif mm in removed_elements:
                new_list_points = [new_list_points[kk] for kk, elem in enumerate(mm_in_sublists) if not elem]
                new_list_costs = [new_list_costs[kk] for kk, elem in enumerate(mm_in_sublists) if not elem]
                new_list_x_est = [new_list_x_est[kk] for kk, elem in enumerate(mm_in_sublists) if not elem]

        # Update possible elements
        elements = elements - set(removed_elements)

    # print('In function')
    # code.interact(local=dict(globals(), **locals()))

    return solution, costs, x_est, new_list_points, new_list_costs, new_list_x_est, removed_elements,

# This function selects from the consistent point permutations, the ones that created the image
def find_possible_solutions(x, list_points, list_costs, x_est_total):
    # Remove duplicate lists of points, which are permutations of each other
    unique_list_points, unique_costs, unique_x_est = remove_duplicates(list_points, list_costs, x_est_total)
    for aa, bb in zip(unique_list_points, unique_x_est):
        if len(aa) != len(bb):
            print('Duplicates')
            code.interact(local=dict(globals(), **locals()))

    if not unique_list_points:
        print('Nothing found')
        return False, [], [], []

    # print('before function')
    # code.interact(local=dict(globals(), **locals()))

    #Identify permutations including a point not appearing anywhere else
    solution, costs, x_est, unique_list_points, unique_costs, unique_x_est, removed_elements \
        = identify_unique_permutations(x, unique_list_points, unique_costs, unique_x_est)
    for aa, bb in zip(unique_list_points, unique_x_est):
        if len(aa) != len(bb):
            print('permutations')
            code.interact(local=dict(globals(), **locals()))

    # print('after function')

    #Sort remaining points by cost
    cheap_to_expensive = np.argsort(unique_costs)
    # code.interact(local=dict(globals(), **locals()))
    costs_sorted = np.array(unique_costs)[cheap_to_expensive].tolist()
    points_sorted = np.array(unique_list_points,dtype=list)[cheap_to_expensive].tolist()
    x_est_sorted = np.array(unique_x_est,dtype=list)[cheap_to_expensive].tolist()

    # print('other code')
    # code.interact(local=dict(globals(), **locals()))

    # Check if the solution is consitent with the image
    if solution:
        points_used = set(reduce(lambda x, y: x + y, solution))
        finished = True if points_used == set(range(x.shape[0])) else False
    else:
        finished = False

    #Select a fixed number of possible candidates to get a solution, increase if not achieved
    n_candidates = 10
    # code.interact(local=dict(globals(), **locals()))
    if unique_list_points:
        solution_indexes = []
        while not solution_indexes and n_candidates <= 500:
            #Create combinations of solutions given the remaining elements in unique_list_points and solution
            solution_indexes, _ = explore_space_solutions(points_sorted[:n_candidates], [], set(removed_elements), 0, np.shape(x)[0], [])
            n_candidates *= 2

        #Collect all solutions and costs, and select the solution with less costs
        if solution_indexes:
            finished = True
            solutions = [solution + np.array(points_sorted,dtype=list)[ind].tolist() for ind in solution_indexes]
            costs = [sum(costs + np.array(costs_sorted)[ind].tolist()) for ind in solution_indexes]
            x_ests = [x_est + np.array(x_est_sorted,dtype=list)[ind].tolist() for ind in solution_indexes]
            solution = solutions[np.argmin(costs)]
            x_est = x_ests[np.argmin(costs)]
        else:
            finished = False
            print('Finding solutions failed')

    return finished, [x[elem] for elem in solution], solution, x_est

# # This function selects from the consistent point permutations, the ones that created the image
# def find_possible_solutions(x, list_points, list_costs):
#     # Remove duplicate lists of points, which are permutations of each other
#     unique_list_points, unique_costs = remove_duplicates(list_points, list_costs)
#
#     #Create triples for easier use (pos,points,costs)
#     data = [(pos,points,costs) for pos,points,costs in zip(range(len(unique_list_points)),unique_list_points,unique_costs)]
#
#     # Find all lists of sublists containing non repeated elements
#     global_solution = []
#     M = np.shape(x)[0]
#     print(len(unique_list_points))
#     _ = test(data, [], global_solution, M, 0)
#
#     # Remove solutions that not contain all the points in the image
#     feasible_solutions = []
#     for pattern in global_solution:
#         used_points = []
#         for _,points,_ in pattern:
#             used_points += points
#         if set(used_points) == set(range(M)):
#             feasible_solutions.append(pattern)
#
#     # Get costs of solutions
#     total_cost = np.inf
#     solution = []
#     for pattern in feasible_solutions:
#         new_cost = 0
#         for _,points,cost in pattern:
#             new_cost += cost
#         if new_cost < total_cost:
#             total_cost = new_cost
#             solution.append(pattern)
#
#     #Return solution
#     if not solution:
#         return False, []
#     else:
#         points = []
#         for elem in solution[0]:
#             points += [elem[1]]
#         return True, [x[elem] for elem in points]


def count_points(list_points, M):
    counts = np.zeros(M, np.int)
    for mm in range(M):
        counts[mm] = sum([mm in sublist for _,sublist,_ in list_points])
    return counts

def remove_used_terms(list_points, sublist):
    solution = []
    for obj in list_points:
        point = obj[1]
        is_there = sum([elem in point for elem in sublist])
        if not is_there:
            solution.append(obj)
    return solution

# Recursive function
def test(data, solution, global_solution,  M, counter):

    points = list(range(M))
    counts = count_points(data, M)
    # Return condition
    if not sum(counts):
        return True

    # Get points with less counts, discarding counts=0.
    counts[counts == 0] = 1000  # Change this
    ind = np.argmin(counts[counts != 0])
    next_point = points[ind]
    #Get candidates
    candidates = [(pos, sublist, cost) for pos, sublist, cost in data if next_point in sublist]

    # if equal to 1, get the sublist
    if len(candidates) == 1:
        candidate = candidates[0]
        new_data = remove_used_terms(data, candidate[1])
        finished = test(new_data, solution + [candidate], global_solution, M, counter + 1)
        if finished:
            global_solution.append(solution + [candidate])

    else:
        # Get candidate sublists
        for candidate in candidates:
            new_data = remove_used_terms(data, candidate[1])
            finished = test(new_data, solution + [candidate], global_solution, M, counter+1)
            if finished:
                global_solution.append(solution + [candidate])

    return False

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



def run(data_model):
    # tol = 1e-4
    tol = 1e-1
    finished = False
    while not finished and tol <= 1:
        print(tol)
        #Get the list of objects consistent with the points in the image
        list_points, list_x, list_costs, x_est_total = get_possible_objects(data_model, tol)
        # for aa, bb in zip(list_points, x_est_total):
        #     if len(aa) != len(bb):
        #         print('First function')
        #         code.interact(local=dict(globals(), **locals()))

        #From all consistent shapes with the image, return the ones that created the image
        finished, X_obj_est, solution, x_est = find_possible_solutions(data_model['X_m'], list_points, list_costs, x_est_total)
        for aa, bb in zip(solution,x_est):
            if len(aa) != len(bb):
                print('Second function')
                code.interact(local=dict(globals(), **locals()))
        tol *= 10

    return X_obj_est, solution, x_est

def save_results(figures_dir, data_model, X_obj_est, X_transformed, assignment):
    # Create a directory if it doesn't exist
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Save data model
    data_file = figures_dir + 'data_model.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(data_model, f, pickle.HIGHEST_PROTOCOL)
    data_file = figures_dir + 'X_obj_est.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(X_obj_est, f, pickle.HIGHEST_PROTOCOL)
    data_file = figures_dir + 'X_transformed.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(X_transformed, f, pickle.HIGHEST_PROTOCOL)
    data_file = figures_dir + 'assignment.pkl'
    with open(data_file, 'wb') as f:
        pickle.dump(assignment, f, pickle.HIGHEST_PROTOCOL)