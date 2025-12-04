from game import gameutils as gu
import time
import math

if __name__ == "__main__":
    start = time.time_ns()
    sum = 0
    for i in range(1000):
        sum += gu.inv_sqrt(i)
    end = time.time_ns()
    print(sum, end - start)

    start = time.time_ns()
    sum = 0
    for i in range(1000):
        sum += math.sqrt(i)
    end = time.time_ns()
    print(sum, end - start)
