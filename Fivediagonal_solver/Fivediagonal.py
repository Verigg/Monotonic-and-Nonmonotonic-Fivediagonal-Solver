import numpy as np
from tabulate import tabulate


# Генерация матрицы
def Generate_Matrix(size, diagonally_dominant=True):
    err = np.full(size, 0.1)
    if diagonally_dominant:
        c = np.random.randint(size + 1, size + 20, (size,))
    else:
        c = np.random.random_sample((size,)) + err

    d = np.random.random_sample((size,)) + err
    e = np.random.random_sample((size,)) + err
    b = np.random.random_sample((size,)) + err
    a = np.random.random_sample((size,)) + err
    f = np.random.randint(size + 1, size + 20, (size,))

    A = np.zeros((size, size))

    A = A + np.diag(c, 0)
    A = A + np.diag(-d[:size - 1], 1)
    A = A + np.diag(e[:size - 2], 2)
    A = A + np.diag(-b[1:], -1)
    A = A + np.diag(a[2:], -2)
    data = [size - 1, c, d, e, b, a, f]

    return (A, data)


def monotonic(data):
    N, c, d, e, b, a, f = data

    alpha = np.zeros(N + 1)
    gamma = np.zeros(N + 2)
    betta = np.zeros(N)
    delta = np.zeros(N + 1)

    # Начальные условия

    alpha[1] = d[0] / c[0]
    gamma[1] = f[0] / c[0]
    betta[1] = e[0] / c[0]
    delta[1] = c[1] - b[1] * alpha[1]

    alpha[2] = (d[1] - betta[1] * b[1]) / delta[1]
    gamma[2] = (f[1] + b[1] * gamma[1]) / delta[1]
    betta[2] = e[1] / delta[1]

    # Рекурсивный поиск коэффицинтов

    for i in range(2, N + 1):

        delta[i] = c[i] - a[i] * betta[i - 1] + alpha[i] * (a[i] * alpha[i - 1] - b[i])
        if (i <= N - 1):
            alpha[i + 1] = (d[i] + betta[i] * (a[i] * alpha[i - 1] - b[i])) / delta[i]
        gamma[i + 1] = (f[i] - a[i] * gamma[i - 1] - gamma[i] * (a[i] * alpha[i - 1] - b[i])) / delta[i]
        if (i <= N - 2):
            betta[i + 1] = e[i] / delta[i]

    # Обратный ход

    y = np.zeros(N + 1)

    y[N] = gamma[N + 1]

    y[N - 1] = alpha[N] * y[N] + gamma[N]

    for i in range(N - 2, -1, -1):
        y[i] = alpha[i + 1] * y[i + 1] - betta[i + 1] * y[i + 2] + gamma[i + 1]

    return y


def non_monotonic(data):
    N, c, d, e, b, a, f = data
    alpha = np.zeros(N + 1)
    betta = np.zeros(N)
    gamma = np.zeros(N + 2)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    eta = np.zeros(N)
    theta = np.zeros(N + 1)

    # Начальные условия
    C = c[0]
    D = d[0]
    B = b[1]
    Q = c[1]
    S = a[2]
    T = b[2]
    R = 0
    A = a[3]
    F = f[0]
    P = f[1]
    G = f[2]
    H = f[3]
    x[0] = 0
    eta[0] = 1

    # Алгоритм
    for i in range(N - 1):
        # a)
        if (np.abs(C) >= np.abs(D)) and (np.abs(C) >= np.abs(e[i])):
            alpha[i + 1] = D / C
            betta[i + 1] = e[i] / C
            gamma[i + 1] = F / C

            C = Q - B * alpha[i + 1]
            D = d[i + 1] - B * betta[i + 1]
            F = P + B * gamma[i + 1]

            B = T - S * alpha[i + 1]
            Q = c[i + 2] - S * betta[i + 1]
            P = G - S * gamma[i + 1]

            if i != N - 2:
                S = A - R * alpha[i + 1]
                T = b[i + 3] - R * betta[i + 1]
                G = H + R * gamma[i + 1]

            if i < N - 3:
                R = 0
                A = a[i + 4]
                H = f[i + 4]

            theta[i + 1] = x[i]
            x[i + 1] = eta[i]
            eta[i + 1] = i + 2
        # b)
        elif (np.abs(D) > np.abs(C)) and (np.abs(D) >= np.abs(e[i])):
            alpha[i + 1] = C / D
            betta[i + 1] = -e[i] / D
            gamma[i + 1] = -F / D

            C = Q * alpha[i + 1] - B
            D = Q * betta[i + 1] + d[i + 1]
            F = P - Q * gamma[i + 1]

            B = T * alpha[i + 1] - S
            Q = c[i + 2] + T * betta[i + 1]
            P = G + T * gamma[i + 1]

            if i != N - 2:
                S = A * alpha[i + 1] - R
                T = b[i + 3] + A * betta[i + 1]
                G = H - A * gamma[i + 1]

            if i < N - 3:
                R = 0
                A = a[i + 4]
                H = f[i + 4]

            theta[i + 1] = eta[i]
            x[i + 1] = x[i]
            eta[i + 1] = i + 2
        # c)
        elif (np.abs(e[i]) > C) and (np.abs(e[i]) > np.abs(D)):
            alpha[i + 1] = D / e[i]
            betta[i + 1] = C / e[i]
            gamma[i + 1] = F / e[i]

            C = Q - d[i + 1] * alpha[i + 1]
            D = B - d[i + 1] * betta[i + 1]
            F = P + d[i + 1] * gamma[i + 1]

            B = T - c[i + 2] * alpha[i + 1]
            Q = S - c[i + 2] * betta[i + 1]
            P = G - c[i + 2] * gamma[i + 1]

            if i != N - 2:
                S = A - b[i + 3] * alpha[i + 1]
                T = R - b[i + 3] * betta[i + 1]
                G = H + b[i + 3] * gamma[i + 1]

            if i < N - 3:
                R = -a[i + 4] * alpha[i + 1]
                A = -a[i + 4] * betta[i + 1]
                H = f[i + 4] - a[i + 4] * gamma[i + 1]

            theta[i + 1] = i + 2
            x[i + 1] = eta[i]
            eta[i + 1] = x[i]

    if np.allclose(np.abs(C), np.abs(D), atol=5) or np.abs(C) > np.abs(D):
        alpha[N] = D / C
        gamma[N] = F / C
        gamma[N + 1] = (P + B * gamma[N]) / (Q - B * alpha[N])
        theta[N] = x[N - 1]
        x[N] = eta[N - 1]
    elif np.abs(D) > np.abs(C):
        alpha[N] = C / D
        gamma[N] = -F / C
        gamma[N + 1] = (P - Q * gamma[N]) / (Q * alpha[N] - B)
        theta[N] = eta[N - 1]
        x[N] = x[N - 1]

    m = int(theta[N])
    n = int(x[N])

    y[n] = gamma[N + 1]
    y[m] = alpha[N] * y[N] + gamma[N]

    for i in reversed(range(N - 1)):
        m = int(theta[i + 1])
        n = int(x[i + 1])
        k = int(eta[i + 1])
        y[m] = alpha[i + 1] * y[n] - betta[i + 1] * y[k] + gamma[i + 1]
    return y


def test_accuracy(result, expected):
    return np.allclose(result, expected, atol=1e-10)


def main():
    def generate_log():
        filename = str("log_seed_" + str(salt) + ".txt")
        log = open(filename, "w")
        log.write("seed:" + str(salt) + "\n")
        log.write("size:" + str(size) + "\n")
        log.write("size:" + str(diagonally_dominant) + "\n")
        log.write(tabulate(A, tablefmt="grid") + "\n")
        log.write("Monotonic:" + "\n")
        log.write("result = " + str(result_Monotonic.tolist()) + "\n")
        log.write("expected = " + str(expected.tolist()) + "\n")
        log.write("Delta = " + str((result_Monotonic - expected).tolist()) + "\n")
        log.write("Nonmonotonic:" + "\n")
        log.write("result = " + str(result_NonMonotonic.tolist()) + "\n")
        log.write("expected = " + str(expected.tolist()) + "\n")
        log.write("Delta = " + str((result_NonMonotonic - expected).tolist()) + "\n")
        log.close()

    salt = 1
    np.random.seed(salt)

    size = 10

    diagonally_dominant = False
    A, data = Generate_Matrix(size, diagonally_dominant)
    expected = np.linalg.solve(A, data[6])
    result_Monotonic = monotonic(data)
    result_NonMonotonic = non_monotonic(data)

    print(tabulate(A, tablefmt="grid"), "\n")

    print("Monotonic:")
    print("result = ", result_Monotonic.tolist())
    print("expected = ", expected.tolist())
    print("Delta = ", (result_Monotonic - expected).tolist(), "\n")

    if test_accuracy(result_Monotonic, expected):
        pass
    else:
        raise "result != expected"

    print("Nonmonotonic:")
    print("result = ", result_NonMonotonic.tolist())
    print("expected = ", expected.tolist())
    print("Delta = ", (result_NonMonotonic - expected).tolist())

    if test_accuracy(result_NonMonotonic, expected):
        pass
    else:
        raise "result != expected"

    generate_log()


if __name__ == "__main__":
    main()

# Тест из книги 1978г 103 стр
# https://samarskii.ru/books/book1978.pdf
# A = np.array([[1, -1, +2, 0, 0, 0, 0, 0, 0, 0, 0],
#             [-1, +1, -1, +1, 0, 0, 0, 0, 0, 0, 0],
#             [1, -1, +2, -1, +1, 0, 0, 0, 0, 0, 0],
#             [0, 1, -1, +2, -1, +1, 0, 0, 0, 0, 0],
#             [0, 0, 1, -1, +2, -1, +1, 0, 0, 0, 0],
#             [0, 0, 0, 1, -1, +2, -1, +1, 0, 0, 0],
#             [0, 0, 0, 0, 1, -1, +2, -1, +1, 0, 0],
#             [0, 0, 0, 0, 0, 1, -1, +2, -1, +1, 0],
#             [0, 0, 0, 0, 0, 0, 1, -1, +2, -1, +1],
#             [0, 0, 0, 0, 0, 0, 0, 1, -1, +1, -1],
#             [0, 0, 0, 0, 0, 0, 0, 0, 2, -1, +1]])
#
# c = [1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1]
# d = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]
# e = [2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
# b = list(reversed(d))
# a = list(reversed(e))
# f = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# size = 11
# data = [size - 1, c, d, e, b, a, f]
