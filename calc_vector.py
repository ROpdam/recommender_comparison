from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np


def calc_vector(calc_user, fixed, m, m_ones, res, name, self_items):
    num_solve = self_items['total_users'] if calc_user else self_items['total_items']
    num_fixed = fixed.shape[0]
    YTY = fixed.T.dot(fixed)
    I_for_C = sparse.eye(num_fixed)  # The 1 in the confidence formula
    lambda_I = self_items['reg'] * sparse.eye(self_items['nolf'])
    solve_vecs = np.zeros([num_solve, self_items['nolf']])

    for i in range(num_solve):
        if calc_user:
            pu = m_ones[i].toarray()
            Cu = self_items['alpha'] * sparse.diags(m[i].toarray(), [0])
        else:
            pu = m_ones[:, i].T.toarray()
            Cu = self_items['alpha'] * sparse.diags(m[:, i].T.toarray(), [0])

        YTCuIY = fixed.T.dot(Cu + I_for_C).dot(fixed)
        YTCupu = fixed.T.dot(Cu).dot(sparse.csr_matrix(pu).T)  # result
        xu = spsolve(YTY + YTCuIY + lambda_I, YTCupu)
        solve_vecs[i] = xu

    res[name] = solve_vecs
    return solve_vecs
