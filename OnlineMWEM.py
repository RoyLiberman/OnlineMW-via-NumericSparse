import numpy as np
import numpy.linalg as LA
import numpy.random as random
from HistogramDataSet import HistogramDataSet

STOP_SIGN = "STOP_SIGN"


def mw(x_i_t, query_t, v_t, eta):
    query_answer_t = x_i_t.evaluate(query_t)
    if v_t < query_answer_t:
        r_t = query_t
    else:
        r_t = 1 - query_t

    x_i_t_plus_1 = x_i_t.histogram * (np.exp(- eta * r_t))
    x_i_t_plus_1 = x_i_t_plus_1 * (1 / np.sum(x_i_t_plus_1))
    return x_i_t_plus_1


def numeric_sparse(data, query_list, T, c, epsilon, delta):
    if delta == 0:
        epsilon_1 = 8 * epsilon / 9
        epsilon_2 = 2 * epsilon / 9

        def sigma(eps):
            return 2 * c / eps
    else:
        epsilon_1 = np.sqrt(512) * epsilon / (np.sqrt(512) + 1)
        epsilon_2 = 2 * epsilon / (np.sqrt(512) + 1)

        def sigma(eps):
            return np.sqrt(32 * c * np.log(2 / delta)) / eps

    T_hat_count = T + random.laplace(scale=sigma(epsilon_1))
    count = 0
    answer_list = []
    for query in query_list:
        v_query_1 = random.laplace(scale=2 * sigma(epsilon_1))
        query_response = data.evaluate(query)
        if query_response + v_query_1 >= T_hat_count:
            v_query_2 = random.laplace(scale=sigma(epsilon_2))
            answer_list.append(query_response + v_query_2)
            count = count + 1
            T_hat_count = T + random.laplace(scale=sigma(epsilon_1))
        else:
            answer_list.append(STOP_SIGN)
        if count >= c:
            return answer_list
    return answer_list


def calculate_threshold(data, query_list, epsilon, delta, alpha, beta, q_class_size, xhi_size):
    c = (4 * np.log2(xhi_size)) / (alpha ** 2)
    k = len(query_list)
    T_denomenator = epsilon * LA.norm(data.histogram, ord=1)
    if delta == 0:
        T_numerator = 18 * c * (np.log2(2 * q_class_size) + np.log2(4 * c / beta))
        T = T_numerator / T_denomenator
    else:
        T_numerator = (2 + 32 * np.sqrt(2)) * (np.sqrt(c * np.log2(2 / delta))) * (np.log2(k) + np.log2(4 * c / beta))
        T = T_numerator / T_denomenator
    return T


def numeric_sparse_online_mw(data, query_list, epsilon, delta, alpha, beta, q_class_size, xhi_size, data_dim, iterations):
    c = (4 * np.log2(xhi_size)) / (alpha ** 2)
    T = calculate_threshold(data, query_list, epsilon, delta, alpha, beta, q_class_size, xhi_size)
    eta = alpha / 2.0
    t = 0
    x_t_histogram_data_set = HistogramDataSet(xhi_size, data_dim, create_random=True)
    while t < iterations:
        for i, query in enumerate(query_list):
            query_answer_on_x_t = x_t_histogram_data_set.evaluate(query)
            altered_answer_a = numeric_sparse(data.subtract(x_t_histogram_data_set), [query], T, c, epsilon, delta)[0]

            altered_answer_b = numeric_sparse(x_t_histogram_data_set.subtract(data), [query], T, c, epsilon, delta)[0]

            if altered_answer_a == STOP_SIGN and altered_answer_b == STOP_SIGN:
                a_i = query_answer_on_x_t
            else:
                if altered_answer_a != STOP_SIGN:
                    a_i = query_answer_on_x_t + altered_answer_a
                else:
                    a_i = query_answer_on_x_t - altered_answer_b
            x_t = mw(x_t_histogram_data_set, query, a_i, eta)
            x_t_histogram_data_set = HistogramDataSet(xhi_size, data_dim, x_t, histogram_data_set=True)
            t = t + 1
    return x_t_histogram_data_set
