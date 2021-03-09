import numpy as np
from scipy.special import digamma, gammaln, logsumexp

# Set the hyperparameters of the model (mu_0, Lambda_0) for Y and lambda_0
def hyperparams_initialization(data_model, mu_0=np.zeros(4), Lambda_0=np.eye(4), lambda_0=10):
    # Prior for Z -> uniform
    mean_x = data_model['X_m'].mean(0)
    mu_0 = np.zeros(4)
    mu_0[:2] = mean_x
    alpha_0 = np.ones(data_model['N'])

    return dict({'lambda_0': lambda_0,
                 'Lambda_0': Lambda_0,
                 'mu_0': mu_0,
                 'alpha_0': alpha_0,
                 })

def params_initialization(r_mnk, data_model, hyper_params):
    K = data_model['K']
    return dict({'alpha_k': [K*[hyper_params['alpha_0']]],
                 'Lambda_k': [K*[hyper_params['Lambda_0']]],
                 'mu_k': [K*[hyper_params['mu_0']]],
                 'r_mnk': [r_mnk],
                 'lambda_0': [hyper_params['lambda_0']],
                 })

# Assignment matrix initialization (random)
def r_mnk_initialization(data_model):
    r_mnk = np.random.rand(data_model['M'], data_model['N'])
    return r_mnk / np.sum(r_mnk, 1).reshape(-1, 1)

def compute_alpha_k(params, hyper_params):

    return hyper_params['alpha_0'] + np.sum(params['r_mnk'][-1], 0)

def compute_Lambda_k(params, hyper_params, data_model):

    r_mnk = params['r_mnk'][-1]
    Lambda_0, lambda_0 = hyper_params['Lambda_0'], hyper_params['lambda_0']
    N_k, K = data_model['N_k'], data_model['K']

    Lambda_k = []
    index_k = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
    for kk in range(K):
        r_k = np.sum(r_mnk[:, index_k[kk]:index_k[kk + 1]], 0)
        Lambda_k += [Lambda_0 + lambda_0*np.sum([r_k[nn] * data_model['FF'][kk][nn] for nn in range(N_k[kk])], 0)]

    return Lambda_k

def compute_mu_k(params, hyper_params, data_model, Lambda_k_inv):

    r_mnk = params['r_mnk'][-1]
    mu_0, Lambda_0, lambda_0 = hyper_params['mu_0'], hyper_params['Lambda_0'], hyper_params['lambda_0']
    N_k, K, M = data_model['N_k'], data_model['K'], data_model['M']

    mu_k = []
    index_k = np.concatenate([np.zeros(1), np.cumsum(N_k)]).astype(int)
    for kk in range(K):
        r_mn = r_mnk[:, index_k[kk]:index_k[kk + 1]]
        F_xm = data_model['F_xm'][kk]

        aux = Lambda_0 @ mu_0 + lambda_0*np.sum([r_mn[mm].reshape(1, -1) @ F_xm[mm] for mm in range(M)], 0)[0]
        mu_k += [Lambda_k_inv[kk] @ aux]

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

def create_dummies(M, N):

    return np.log(N)*np.ones([N - M, N])

def sinkhorn_klopp(log_r):

    M, N = np.shape(log_r)
    sum_cols = logsumexp(log_r, axis=1)
    counter = 0
    while not (np.abs(np.exp(sum_cols[:M]) - 1) < 1e-3).all():
        log_r = log_r - logsumexp(log_r, axis=0).reshape(1, -1)
        sum_cols = logsumexp(log_r, axis=1)
        log_r = log_r - sum_cols.reshape(-1, 1)
        counter += 1

    return log_r

def compute_r_mnk(params, hyper_params, data_model, mahal_term, trace_term, normalize_rows=True):

    alpha_k = params['alpha_k'][-1]
    lambda_0 = hyper_params['lambda_0']
    M, N = data_model['M'], data_model['N']

    #E[log pi]
    E_log_pi = digamma(alpha_k) - digamma(np.sum(alpha_k))

    log_rho = -np.log(2 * np.pi) + np.log(lambda_0) + E_log_pi.reshape(1, -1) - 0.5 * lambda_0 * (
        mahal_term + np.reshape(trace_term, [1, -1]))

    if normalize_rows:
        #Check if dummy variables are needed
        if M < N:
            dummies = create_dummies(M, N)
            log_rho = np.concatenate([log_rho, dummies], 0)
        log_r = sinkhorn_klopp(log_rho)
        return np.exp(log_r[:M,:]) #Remove dummy rows
    else:
        #Only make rows sum to 1
        log_r = log_rho - logsumexp(log_rho, axis=1).reshape(-1, 1)
        return np.exp(log_r)

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

def compute_KL_Z(params):

    r_mnk, alpha_k = params['r_mnk'][-1], params['alpha_k'][-1]

    return np.sum(r_mnk * (digamma(alpha_k) - digamma(np.sum(alpha_k.reshape([1, -1]))) - np.log(r_mnk + 1e-9)))

def compute_KL_pi(params, hyper_params):

    alpha_k = params['alpha_k'][-1]
    alpha_0 = hyper_params['alpha_0']

    E_log_pi = digamma(alpha_k) - digamma(np.sum(alpha_k))
    log_beta_alpha_0 = np.sum(gammaln(alpha_0)) - gammaln(np.sum(alpha_0))
    log_beta_alpha_k = np.sum(gammaln(alpha_k)) - gammaln(np.sum(alpha_k))

    return np.sum((alpha_0 - alpha_k) * E_log_pi) - log_beta_alpha_0 + log_beta_alpha_k

def lambda_MAP_estimation(params, data_model, mahal_term, trace_term):

    r_mnk = params['r_mnk'][-1]
    M = data_model['M']

    mahal_part = np.sum(r_mnk * mahal_term)
    trace_part = np.sum(np.sum(r_mnk, 0) * np.reshape(trace_term, [1, -1]))

    return M/(1e-3+0.5*(mahal_part + trace_part))


def params_update(data_model, params, hyper_params):

    # Update equations - alpha
    params['alpha_k'].append(compute_alpha_k(params, hyper_params))

    # Update equations - Lambda_k
    params['Lambda_k'].append(compute_Lambda_k(params, hyper_params, data_model))

    # Update equations - mu_k
    Lambda_k_inv = np.linalg.inv(params['Lambda_k'][-1])
    params['mu_k'].append(compute_mu_k(params, hyper_params, data_model, Lambda_k_inv))

    # Compute statistics
    mahal_term = compute_mahal_term(params, data_model)
    muk_mu0_term = compute_muk_mu0_mahal(params, hyper_params, data_model)
    trace_term = compute_trace_term(Lambda_k_inv, data_model)

    # Update r_mnk
    params['r_mnk'].append(compute_r_mnk(params, hyper_params, data_model, mahal_term, trace_term, False))

    # Save current lambda
    params['lambda_0'].append(hyper_params['lambda_0'])

    #Compute ELBO
    E_log_p_x = compute_E_log_p_x(params, hyper_params, data_model, mahal_term, trace_term)
    KL_y = compute_KL_Y(params, hyper_params, data_model, Lambda_k_inv, muk_mu0_term)
    KL_z = compute_KL_Z(params)
    KL_pi = compute_KL_pi(params, hyper_params)
    score = np.sum(mahal_term * params['r_mnk'][-1])

    ELBO_terms = [E_log_p_x, KL_y, KL_z, KL_pi]

    return params, ELBO_terms, score

def stop(ELBO_epoch, new_ELBO, hyper_params, params, data_model, lambda_0=10):

    ELBO = ELBO_epoch[-1]
    return True if np.abs(ELBO - new_ELBO) < 1e-3 else False