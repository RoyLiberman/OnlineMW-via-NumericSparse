import numpy as np
import matplotlib.pyplot as plt
from HistogramDataSet import HistogramDataSet
from OnlineMWEM import numeric_sparse_online_mw


def analyze_results(n, data_dimension, epsilon, delta, alpha, beta, iterations):

    q_class_size = 2 ** data_dimension
    xhi_size = 2 ** data_dimension

    # random binary queries
    query_list = [
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size)
    ]

    # random binary tabular data set
    data = np.random.randint(2, size=(n, data_dimension))
    hist_data = HistogramDataSet(xhi_size, data_dimension, data, tabular_data_set=True)
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


def main():
    data_dimension = 5
    n = 10
    epsilon = 0.01
    delta = 0.5
    alpha = 0.1
    beta = 0.1
    iterations = 20000
    analyze_results(n, data_dimension, epsilon, delta, alpha, beta, iterations)


if __name__ == "__main__":
    main()
