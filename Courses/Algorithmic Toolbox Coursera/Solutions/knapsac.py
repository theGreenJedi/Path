# Uses python3
import sys


def optimal_weight(W, w):
    """Knapsack without repetitions.
    It's given a set of bars of gold and the goal is to take as much gold as
    possible into the bag. There is just one copy of each bar and for each bar
    you can either take it or not (hence you cannot take a fraction of a bar).
    Samples:
    >>> optimal_weight(10, [1, 4, 8])
    9
    >>> # Explanation:
    >>> # The sum of the weights of the first and the last bar is equal to 9.
    >>> #
    >>> # Weight matrix:
    >>> #
    >>> #                  W(i)
    >>> #          0 1 2 3 4 5 6 7 8 9 10
    >>> #        +-----------------------
    >>> #        | 0 0 0 0 0 0 0 0 0 0 0
    >>> #      1 | 0 1 1 1 1 1 1 1 1 1 1
    >>> # w(j) 4 | 0 1 1 1 1 5 5 5 5 5 5
    >>> #      8 | 0 1 1 1 1 5 5 5 5 9 9
    >>> #        +-----------------------
    """
    w = [0] + w
    items = len(w)
    capacity = W + 1

    # Create a weight matrix and write in initial values.
    weights = [[0 for _ in range(items)] for _ in range(capacity)]

    for j in range(1, items):
        for i in range(1, capacity):
            prev = weights[i][j - 1]
            cur = w[j] + weights[W - w[j]][j - 1]
            if cur > i:
                weights[i][j] = prev
            else:
                weights[i][j] = max(prev, cur)

    return weights[-1][-1]


if __name__ == "__main__":
    input = sys.stdin.read()
    W, n, *w = list(map(int, input.split()))
print(optimal_weight(W, w))
