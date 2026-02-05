import numpy as np


def nearest_neighbor(dist: np.ndarray, running_dist: int = 0, path: list[int] = None) -> tuple[bool, int, list[int]]:
    if path is None:
        # Initializing
        path = [1]
    # If we reached the destination, done!
    if path[-1] == 10:
        return True, running_dist, path
    # Use the nearest neighbor
    choices = dist[path[-1],:]
    # Argsort with nan last (those are not options
    argidx = np.argsort(choices)
    for idx in argidx:
        if not np.isnan(choices[idx]) and idx not in path:
            try_pass, try_dist, try_path = nearest_neighbor(dist,running_dist+choices[idx], path + [idx])
            if try_pass:
                return True, try_dist, try_path
    # This is exhaustive.
    return False,-1,[]


def dijkstra(dist: np.ndarray) -> tuple[bool, int, list[int]]:
    visit_dist = [0]*(len(dist)+1)
    visit_path = [[]]*len(dist)
    visit_path[1] = [1]
    # Start at 1
    for cur_city in range(1,11):
        for other_city in range(1,11):
            if cur_city != other_city and not np.isnan(dist[cur_city, other_city]):
                # Check if this distance is less
                test_dist = visit_dist[cur_city] + dist[cur_city, other_city]
                if test_dist < visit_dist[other_city] or visit_dist[other_city] == 0:
                    visit_dist[other_city] = test_dist
                    visit_path[other_city] = visit_path[cur_city] + [other_city]
    return True, visit_dist[10], visit_path[10]


def main():
    dist = np.nan *np.ones((11,11))
    dist[1,2] = 16
    dist[1,3] = 10
    dist[1,4] = 25
    dist[2,3] = 20
    dist[2,4] = 3
    dist[2,5] = 15
    dist[3,4] = 40
    dist[3,6] = 6
    dist[4,5] = 11
    dist[4,6] = 32
    dist[4,7] = 23
    dist[5,6] = 22
    dist[5,7] = 27
    dist[5,8] = 19
    dist[6,7] = 16
    dist[6,9] = 28
    dist[7,8] = 30
    dist[7,9] = 10
    dist[8,9] = 25
    dist[8,10] = 14
    dist[9,10] = 9

    # Dijkstra's algorithm
    passed, min_len, min_path = dijkstra(dist)
    print(f"Dijkstra's={passed}, min_len={min_len}, min_path={min_path}")
    # Nearest neighbor algorithm
    passed, min_len, min_path = nearest_neighbor(dist)
    print(f"Nearest Neighbor={passed}, dist={min_len}, path={min_path}")



if __name__ == "__main__":
    main()