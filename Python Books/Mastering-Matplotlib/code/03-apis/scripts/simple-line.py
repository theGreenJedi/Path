#! /usr/bin/env python3.4
import matplotlib.pyplot as plt

def main ():
    plt.plot([1,2,3,4])
    plt.ylabel('some numbers')
    plt.savefig("simple-line.png")

if __name__ == "__main__":
    main()
