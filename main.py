import numpy as np
import numpy.linalg as LA
import numpy.random as random
import matplotlib.pyplot as plt
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
        epsilon_2 = 2 / (np.sqrt(512) + 1)

        def sigma(eps):
            return np.sqrt(32 * c * np.log(2 / delta)) / eps

    T_hat_count = T + random.laplace(sigma(epsilon_1))
    count = 0
    answer_list = []
    for query in query_list:
        v_query_1 = random.laplace(2 * sigma(epsilon_1))
        query_response = data.evaluate(query)
        if query_response + v_query_1 >= T_hat_count:
            v_query_2 = random.laplace(sigma(epsilon_2))
            answer_list.append(query_response + v_query_2)
            count = count + 1
            T_hat_count = T + random.laplace(sigma(epsilon_1))
        else:
            answer_list.append(STOP_SIGN)
        if count >= c:
            return answer_list
    return answer_list


def numeric_sparse_online_mw(data, query_list, epsilon, delta, alpha, beta, q_class_size, xhi_size, data_dim, iterations):
    c = (4 * np.log(xhi_size)) / (alpha ** 2)
    k = len(query_list)
    eta = alpha / 2.0
    T_denomenator = epsilon * LA.norm(data.histogram, ord=1)
    if delta == 0:
        T_numerator = 18 * c * (np.log(2 * q_class_size) + np.log(4 * c / beta))
        T = T_numerator / T_denomenator
    else:
        T_numerator = (2 + 32 * np.sqrt(2)) * (np.sqrt(c * np.log(2 / delta))) * (np.log(k) + np.log(4 * c / beta))
        T = T_numerator / T_denomenator

    t = 0
    x_t_histogram_data_set = HistogramDataSet(xhi_size, data_dim, create_random=True)
    while t < iterations:
        for i, query in enumerate(query_list):
            query_answer_on_x_t = x_t_histogram_data_set.evaluate(query)

            altered_query_a = query - query_answer_on_x_t
            altered_answer_a = numeric_sparse(data, [altered_query_a], T, c, epsilon, delta)[0]

            altered_query_b = query_answer_on_x_t - query
            altered_answer_b = numeric_sparse(data, [altered_query_b], T, c, epsilon, delta)[0]

            if altered_answer_b == STOP_SIGN and altered_answer_b == STOP_SIGN:
                a_i = query_answer_on_x_t
            else:
                if altered_answer_a:
                    a_i = query_answer_on_x_t + altered_answer_a
                else:
                    a_i = query_answer_on_x_t - altered_answer_b
            x_t = mw(x_t_histogram_data_set, query, a_i, eta)
            x_t_histogram_data_set = HistogramDataSet(xhi_size, data_dim, x_t, histogram_data_set=True)
            t = t + 1
    return x_t_histogram_data_set


def main():
    data_dimension = 5
    n = 10
    epsilon = 1
    delta = 0
    alpha = 0.1
    beta = 0.1
    q_class_size = 2 ** data_dimension
    xhi_size = 2 ** data_dimension
    iterations = 20000
    query_list = [
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size)
    ]


    data = np.random.randint(2, size=(n, data_dimension))
    hist_data = HistogramDataSet(xhi_size, data_dimension, data, tabular_data_set=True)

    if delta == 0:
        alpha_theoretic_bound_up = 32 * np.log2(xhi_size) * (np.log2(q_class_size) + np.log2(32 * np.log2(xhi_size) / ((alpha ** 2) * beta)))
        alpha_theoretic_bound_down = epsilon * (alpha ** 2) * LA.norm(hist_data.histogram, ord=1)
        alpha_theoretic_bound = alpha_theoretic_bound_up / float(alpha_theoretic_bound_down)
    else:
        alpha_theoretic_bound_up = (2 + 32 * np.sqrt(2)) * np.sqrt(np.log2(xhi_size) * np.log2(2 / delta)) * (np.log2(q_class_size) + np.log2((32 * np.log2(xhi_size)) / ((alpha ** 2) * beta)))
        alpha_theoretic_bound_down = epsilon * alpha * LA.norm(hist_data.histogram, ord=1)
        alpha_theoretic_bound = alpha_theoretic_bound_up / float(alpha_theoretic_bound_down)

    synthetic_data = numeric_sparse_online_mw(hist_data, query_list, epsilon, delta, alpha, beta, q_class_size, xhi_size, data_dimension, iterations)


    diff_list = []
    for i in range(len(query_list)):
        answer_for_query_real = hist_data.evaluate(query_list[i])
        answer_for_query_synthetic = synthetic_data.evaluate(query_list[i])
        diff = np.abs(answer_for_query_real - answer_for_query_synthetic)
        diff_list.append(diff)
        print(f"true query answer for query {i}: {answer_for_query_real}")
        print(f"synthetic query answer {i}: {answer_for_query_synthetic}")
        print(f"diff: {diff}")
    print(f"diff avg: {np.mean(diff_list)}")
    print(f"diff std: {np.std(diff_list)}")



    plt.hist(hist_data.histogram, len(hist_data.histogram))
    plt.show()
    plt.figure()

    plt.hist(synthetic_data.histogram, len(synthetic_data.histogram))
    plt.show()




main()
