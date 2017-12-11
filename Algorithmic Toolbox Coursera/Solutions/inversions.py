# Uses python3
import sys


def merge(b, c):
    """Merge procedure.
    Returns the resulting sorted array and the number inversions.
    An inversion of a sequence a[0], a[1], ... , a[n−1] is a pair of indices
    0 ≤ i < j < n such that a[i] > a[j] . The number of inversions of a sequence
    in some sense measures how close the sequence is to being sorted.
    Samples:
    >>> b = [1, 2, 3, 4, 5]
    >>> c = [3, 5, 6, 8, 9]
    >>> merge(b, c)
    ([1, 2, 3, 3, 4, 5, 5, 6, 8, 9], 2)
    """
    result = []
    inversions = 0
    while b and c:
        if b[0] <= c[0]:
            result.append(b.pop(0))
        else:
            result.append(c.pop(0))
            inversions += len(b)

    result += b or c
    return result, inversions


def merge_sort(a):
    """Implementation of merge sort algorithm.
    Returns a sorted array A and the number of inversions in A.
    Samples:
    >>> a = [2, 3, 9, 2, 9]
    >>> merge_sort(a)
    ([2, 2, 3, 9, 9], 2)
    """
    if len(a) == 1:
        return a, 0

    mid = len(a) // 2
    left, left_inv = merge_sort(a[:mid])
    right, right_inv = merge_sort(a[mid:])

    merged, merged_inv = merge(left, right)
    return merged, merged_inv + left_inv + right_inv


if __name__ == "__main__":
    input = sys.stdin.read()
    n, *a = list(map(int, input.split()))
print(merge_sort(a)[1])
