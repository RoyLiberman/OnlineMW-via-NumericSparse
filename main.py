import numpy as np
import matplotlib.pyplot as plt
from HistogramDataSet import HistogramDataSet
from OnlineMWEM import numeric_sparse_online_mw
import seaborn as sns


def create_heat_map(score_matrix, title, x_axis, y_axis):
    plt.figure(figsize=(12, 12))
    plt.xticks(x_axis)
    plt.yticks(y_axis)
    cluster_ax = sns.heatmap(score_matrix,
                             cmap= sns.light_palette("green"),
                             annot=True,
                             xticklabels=True,
                             yticklabels=True,
                             square=True
                             )
    cluster_ax.set_title(title)
    cluster_ax.set_yticklabels(rotation=0, labels=y_axis)
    cluster_ax.set_xticklabels(rotation=0, labels=x_axis)
    plt.savefig(f"figures/{title}.png")
    plt.close()


def analyze_results(data, query_list, q_class_size, xhi_size, data_dimension, epsilon, delta, alpha, beta, iterations):
    title = f"---------------------- online MWEM with epsilon: {epsilon}, delta: {delta}, alpha: {alpha}, beta: {beta}: ------------------------"
    print(title)
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
    diff_mean = np.mean(diff_list)
    diff_std = np.std(diff_list)
    print(f"diff avg: {diff_mean}")
    print(f"diff std: {diff_std}")
    return diff_mean


def main():
    data_dimension = 10
    n = 30

    epsilon_list = [0.001, 0.01, 0.1, 1]
    delta_list = [0, 0.01, 0.1, 1]
    alpha_list = [1000]
    beta_list = [0.2, 0.5, 1]

    iterations = 10000

    q_class_size = 2 ** data_dimension
    xhi_size = 2 ** data_dimension

    stability_iter = 3

    # random binary queries
    query_list = [
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size),
        np.random.randint(2, size=xhi_size)
    ]

    # random binary tabular data set
    data = np.random.randint(2, size=(n, data_dimension))
    for alpha in alpha_list:
        for beta in beta_list:
            avg_diff_estimation_matrix = np.zeros((len(epsilon_list), len(delta_list)))
            for _ in range(stability_iter):
                for i, epsilon in enumerate(epsilon_list):
                    for j, delta in enumerate(delta_list):
                        avg_diff = analyze_results(data, query_list, q_class_size, xhi_size, data_dimension, epsilon, delta, alpha, beta, iterations)
                        avg_diff_estimation_matrix[i, j] += avg_diff
            plt_title = f"===== alpha: {alpha} beta: {beta} ======="
            create_heat_map(avg_diff_estimation_matrix / stability_iter, plt_title, epsilon_list, delta_list)


if __name__ == "__main__":
    main()
