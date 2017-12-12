# Uses python3
import sys

# Uses python3
def gcd_euclid(a, b):

    remainder = a % b

    if a == 1 or b == 1:
        return 1

    elif remainder == 1:
        return remainder

    elif remainder == 0:
        return b

    else:
         return gcd_euclid(b, remainder)


def lcm_naive(a, b):

    if a < b:
        alpha = b
        beta = a
    else:
        alpha = a
        beta = b

    gcd = gcd_euclid(alpha, beta)

    lcm = (alpha * beta) // gcd

    return int(lcm)

if __name__ == '__main__':
    input = sys.stdin.read()
    a, b = map(int, input.split())
print(lcm_naive(a, b))
