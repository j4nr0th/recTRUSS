import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # A = np.array(
    #     [
    #         [1, 1, 1, 1, 1],
    #         [0, 1, -1, -2, -3],
    #         [0, 1, 1, 4, 9],
    #         [0, 1, -1, -8, -27],
    #         [0, 1, 1, 16, 81]
    #     ])
    # y = np.array([0, 0, 2, 0, 0])
    # print(np.linalg.solve(A, y) * 12)
    # quit()

    DT = 0.01
    t_array = np.arange(0, 10, DT)
    N = len(t_array)
    y_array = np.ones(N)
    u_array = np.zeros(N)
    K = 1
    M = 0.1


    def forcing_term(y, t):
        return - M * np.pi ** 2 * np.cos(np.pi * t)
    # y_computed = - 1 / ((2 * np.pi) ** 2) * np.cos(2 * np.pi * t_array) + (1 / ((2 * np.pi) ** 2))
    # y_array[0:4] = y_computed[0:4]

    for i in range(4, N):
        y_0 = y_array[i - 1]
        y_1 = y_array[i - 2]
        u_0 = forcing_term(y_array[3:i], t_array[i])
        u_array[i] = u_0
        y_approx = 2 * y_0 - y_1 + DT ** 2 / M * u_0
        y_array[i] = y_approx

    # plt.plot(t_array, y_computed, label="analytical")
    plt.plot(t_array, y_array, label="numerical")
    plt.plot(t_array, np.cos(np.pi * (t_array - 3 * DT)))
    plt.plot(t_array, u_array)
    plt.plot(t_array, - np.pi ** 2 * M * np.cos(np.pi * t_array))
    plt.legend()
    plt.show()
