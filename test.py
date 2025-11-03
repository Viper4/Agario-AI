import math

if __name__ == "__main__":
    N = 50
    new_N = 0
    new_population = []
    num_pairs = N // 2
    for j in range(num_pairs):
        h = 4  # Max children
        m = h / num_pairs  # Slope

        children = int(h - m * j)
        new_population.append(children)
        new_N += children
        #print(f"Pair {j}: {children}")
    print(new_population)
    print(new_N)

    new_N = 0
    new_population.clear()
    for i in range(0, N, 2):
        if i + 1 >= N:
            break
        # Reproduce in pairs: 0: (0,1), 1: (2,3),..., floor(n/4): (floor(n/2)-1,floor(n/2))

        # Linear allocation of children based on rank
        parent1 = i
        parent2 = i+1
        h = 2  # Max children
        m = h / N  # Slope

        children = round(h - m * parent1 + h - m * parent2)
        new_population.append(children)
        new_N += children
        #print(f"Pair {i//2} ({i}, {i+1}): {children}")

    print(new_population)
    print(new_N)
