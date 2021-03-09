import numpy as np
from scipy.special import logsumexp
import code
# code.interact(local=dict(globals(), **locals()))

# Set the hyperparameters of the model (mu_0, Lambda_0) for Y and lambda_0
def hyperparams_initialization(data_model, lambda_0=10):
    mu_0 = np.zeros(4)
    Lambda_0 = np.eye(4)
    a_mnk = np.ones([data_model['N'], data_model['N']])/data_model['N']
    return dict({'lambda_0': lambda_0,
                 'Lambda_0': Lambda_0,
                 'mu_0': mu_0,
                 'a_mnk': a_mnk,
                 })

# Initialize dict with the parameters that are neccesary each iteration
def params_initialization(r_mnk, data_model, hyper_params):
    K = data_model['K']
    return dict({'Lambda_k': [K*[hyper_params['Lambda_0']]],
                 'mu_k': [K*[hyper_params['mu_0']]],
                 'r_mnk': [r_mnk],
                 'r_mnk_complete': [r_mnk],
                 'lambda_0': [hyper_params['lambda_0']],
                 })

# Assignment matrix initialization (random)
def r_mnk_initialization(data_model):
    r_mnk = np.random.rand(data_model['M'], data_model['N'])
    return r_mnk / np.sum(r_mnk, 1).reshape(-1, 1)

# Update of Lambda_k
def update_Lambda_k(params, hyper_params, data_model):

    r_mnk = params['r_mnk'][-1]
    Lambda_0, lambda_0 = hyper_params['Lambda_0'], hyper_params['lambda_0']
    N_k, K = data_model['N_k'], data_model['K']

    Lambda_k = []
    index_k = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
    for kk in range(K):
        r_k = np.sum(r_mnk[:, index_k[kk]:index_k[kk + 1]], 0)
        Lambda_k += [Lambda_0 + lambda_0*np.sum([r_k[nn] * data_model['FF'][kk][nn] for nn in range(N_k[kk])], 0)]

    return Lambda_k

# Update of mu_k
def update_mu_k(params, hyper_params, data_model, Lambda_k_inv):

    r_mnk = params['r_mnk'][-1]
    mu_0, Lambda_0, lambda_0 = hyper_params['mu_0'], hyper_params['Lambda_0'], hyper_params['lambda_0']
    N_k, K, M = data_model['N_k'], data_model['K'], data_model['M']

    mu_k = []
    index_k = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
    for kk in range(K):
        r_mn = r_mnk[:, index_k[kk]:index_k[kk + 1]]
        F_xm = data_model['F_xm'][kk]

        aux = Lambda_0 @ mu_0 + lambda_0 * np.sum([r_mn[mm] @ F_xm[mm] for mm in range(M)], 0)
        mu_k += [Lambda_k_inv[kk] @ aux]



    # code.interact(local=dict(globals(), **locals()))

    return mu_k

def compute_muk_mu0_mahal(params, hyper_params, data_model):

    mu_k = params['mu_k'][-1]
    mu_0, Lambda_0 = hyper_params['mu_0'], hyper_params['Lambda_0']
    K = data_model['K']

    return np.sum([(mu_k[kk] - mu_0) @ Lambda_0 @ (mu_k[kk] - mu_0) for kk in range(K)])

def compute_mahal_term(params, data_model):

    mu_k = params['mu_k'][-1]
    N, K, M = data_model['N'], data_model['K'], data_model['M']
    recon_term = np.concatenate([data_model['F'][kk] @ mu_k[kk] for kk in range(K)])

    mahal_term = np.zeros([M, N])
    for mm in range(M):
        mahal_term[mm, :] = np.sum((data_model['X_m'][mm] - recon_term) ** 2, 1)

    return mahal_term

def compute_trace_term(Lambda_k_inv, data_model):

    trace_term = []
    for kk in range(data_model['K']):
        product = data_model['FF'][kk] @ Lambda_k_inv[kk]
        trace_term += [np.trace(elem) for elem in product]

    return trace_term

# Sinkhorn-Knopp algorithm
def sinkhorn_knopp(log_r, max_iters=20, tol=1e-3):

    M, N = np.shape(log_r)
    sum_cols = logsumexp(log_r, axis=1)
    counter = 0
    while counter < max_iters and not (np.abs(np.exp(sum_cols[:M]) - 1) < tol).all():
        log_r = log_r - logsumexp(log_r, axis=0).reshape(1, -1)
        sum_cols = logsumexp(log_r, axis=1)
        log_r = log_r - sum_cols.reshape(-1, 1)
        counter += 1

    return log_r

def update_r_mnk(hyper_params, data_model, mahal_term, trace_term):

    lambda_0 = hyper_params['lambda_0']
    a_mnk = hyper_params['a_mnk']
    M, N = data_model['M'], data_model['N']

    #log_rho is log a_mnk for M = N
    log_rho = np.log(a_mnk)

    #Update log_rho for the points present in the data
    log_rho[:M] -= 0.5 * lambda_0 * (mahal_term + np.reshape(trace_term, [1, -1]))
    log_r = sinkhorn_knopp(log_rho, max_iters=2)

    return np.exp(log_r[:M, :]), np.exp(log_r)

def compute_E_log_p_x(params, hyper_params, data_model, mahal_term, trace_term):

    r_mnk = params['r_mnk'][-1]
    lambda_0 = hyper_params['lambda_0']
    M = data_model['M']

    mahal_part = np.sum(r_mnk * mahal_term)
    trace_part = np.sum(np.sum(r_mnk, 0) * np.reshape(trace_term, [1, -1]))

    return M * (-np.log(2 * np.pi) + np.log(lambda_0)) - 0.5 * lambda_0 * (mahal_part + trace_part)

def compute_KL_Y(params, hyper_params, data_model, Lambda_k_inv, muk_mu0_term):

    Lambda_k = params['Lambda_k'][-1]
    Lambda_0 = hyper_params['Lambda_0']
    K = data_model['K']
    _, dim_y = np.shape(data_model['F'][0][0])

    Lambda_part = K * np.log(np.linalg.det(Lambda_0)) - np.sum(np.log(np.linalg.det(Lambda_k)))
    trace_part = np.sum([np.trace(Lambda_0 @ Lambda_k_inv[kk]) for kk in range(K)])

    return dim_y / 2 * K + 0.5 * Lambda_part - 0.5 * muk_mu0_term - 0.5 * trace_part

def compute_KL_Z(params, hyper_params):

    a_mnk = hyper_params['a_mnk']
    r_mnk_complete = params['r_mnk_complete'][-1]
    # r_mnk, alpha_k = params['r_mnk'][-1], params['alpha_k'][-1]

    return np.sum(r_mnk_complete * (np.log(a_mnk) - np.log(r_mnk_complete + 1e-9)))

# Updates of all parameters of the model
def params_update(data_model, params, hyper_params):

    # Update Lambda_k
    params['Lambda_k'].append(update_Lambda_k(params, hyper_params, data_model))

    # Update mu_k
    Lambda_k_inv = np.linalg.inv(params['Lambda_k'][-1])
    params['mu_k'].append(update_mu_k(params, hyper_params, data_model, Lambda_k_inv))

    # Compute statistics
    mahal_term = compute_mahal_term(params, data_model)
    muk_mu0_term = compute_muk_mu0_mahal(params, hyper_params, data_model)
    trace_term = compute_trace_term(Lambda_k_inv, data_model)

    # Update r_mnk
    r_mnk, r_mnk_complete = update_r_mnk(hyper_params, data_model, mahal_term, trace_term)
    params['r_mnk'].append(r_mnk)
    params['r_mnk_complete'].append(r_mnk_complete)

    # Save current lambda
    params['lambda_0'].append(hyper_params['lambda_0'])

    #Compute ELBO
    E_log_p_x = compute_E_log_p_x(params, hyper_params, data_model, mahal_term, trace_term)
    KL_y = compute_KL_Y(params, hyper_params, data_model, Lambda_k_inv, muk_mu0_term)
    KL_z = compute_KL_Z(params, hyper_params)
    score = np.sum(mahal_term * params['r_mnk'][-1])

    return params, [E_log_p_x, KL_y, KL_z], score

def reinitialize(params, data_model, hyper_params, ELBO_epoch, lambda_0):

    K = data_model['K']
    r_mnk = r_mnk_initialization(data_model)
    params['r_mnk'][-1] = r_mnk
    params['r_mnk_complete'][-1] = r_mnk
    params['mu_k'][-1] = K*[hyper_params['mu_0']]
    params['Lambda_k'][-1] = K*[hyper_params['Lambda_0']]
    params['lambda_0'][-1] = lambda_0
    ELBO_epoch[-1] = -np.inf

def stop(ELBO_epoch, new_ELBO, hyper_params, params, data_model, lambda_0=10):

    lambda_max = 1e4
    tol = 1e-3
    ELBO = ELBO_epoch[-1]

    # code.interact(local=dict(globals(), **locals()))
    N_k, K = data_model['N_k'], data_model['K']
    index_k = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
    if np.abs(ELBO - new_ELBO) < tol:
        # Sparsity Constraint: reset assignments if object assigned with less than 2 points
        for kk in range(K):
            perm_matrix = params['r_mnk'][-1][:,index_k[kk]:index_k[kk+1]]
            if perm_matrix.max() > 0.9 and perm_matrix.sum() < 2:
                # code.interact(local=dict(globals(), **locals()))
                hyper_params['lambda_0'] = lambda_0
                reinitialize(params, data_model, hyper_params, ELBO_epoch, lambda_0)
                return False

        #Continue if lambda hasn't reached its max value
        if hyper_params['lambda_0'] < lambda_max:
            hyper_params['lambda_0'] *= 2
            return False

        # if params['r_mnk'][-1][:,:4].max() > 0.9 and params['r_mnk'][-1][:,:4].sum() < 2:
        #     hyper_params['lambda_0'] = lambda_0
        #     reinitialize(params, data_model, hyper_params, ELBO_epoch, lambda_0)
        #     # code.interact(local=dict(globals(), **locals()))
        #     return False
        # elif params['r_mnk'][-1][:,4:7].max() > 0.9 and params['r_mnk'][-1][:,4:7].sum() < 2:
        #     hyper_params['lambda_0'] = lambda_0
        #     reinitialize(params, data_model, hyper_params, ELBO_epoch, lambda_0)
        #     # code.interact(local=dict(globals(), **locals()))
        #     return False
        # elif params['r_mnk'][-1][:,7:].max() > 0.9 and params['r_mnk'][-1][:,7:].sum() < 2:
        #     hyper_params['lambda_0'] = lambda_0
        #     reinitialize(params, data_model, hyper_params, ELBO_epoch, lambda_0)
        #     # code.interact(local=dict(globals(), **locals()))
        #     return False
        # elif hyper_params['lambda_0'] < lambda_max:
        #     hyper_params['lambda_0'] *= 2
        #     return False
        # else:
        #     return True
    else:
        return False