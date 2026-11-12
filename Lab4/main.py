import math
import random
import time
from multiprocessing import Pool, cpu_count


def calculate_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def total_route_distance(route, cities):
    distance = 0
    num_cities = len(route)
    for i in range(num_cities):
        distance += calculate_distance(cities[route[i - 1]], cities[route[i]])
    return distance


_worker_cities = None


def _init_worker(cities):
    global _worker_cities
    _worker_cities = cities


def evaluate_moves(route, moves, tabu_moves, best_distance, cities=None):
    if cities is None:
        cities = _worker_cities

    best_move = None
    best_move_distance = float("inf")

    for move in moves:
        i, j = move
        candidate_route = list(route)
        candidate_route[i], candidate_route[j] = candidate_route[j], candidate_route[i]
        candidate_distance = total_route_distance(candidate_route, cities)

        if move in tabu_moves and candidate_distance >= best_distance:
            continue
        if candidate_distance < best_move_distance:
            best_move = move
            best_move_distance = candidate_distance

    return best_move, best_move_distance


def split_into_chunks(items, num_chunks):
    chunk_size = math.ceil(len(items) / num_chunks)
    return [items[k:k + chunk_size] for k in range(0, len(items), chunk_size)]


def tabu_search(
    cities,
    max_iterations=1000,
    time_limit=None,
    tabu_tenure=25,
    num_workers=None,
):
    if num_workers is None:
        num_workers = cpu_count()

    num_cities = len(cities)
    results = []

    current_route = list(range(num_cities))
    random.shuffle(current_route)
    current_distance = total_route_distance(current_route, cities)

    best_route = list(current_route)
    best_distance = current_distance

    all_moves = [(i, j) for i in range(num_cities) for j in range(i + 1, num_cities)]
    chunks = split_into_chunks(all_moves, num_workers)

    tabu_list = {}
    pool = Pool(num_workers, initializer=_init_worker, initargs=(cities,)) if num_workers > 1 else None

    start_time = time.time()
    it = 0
    try:
        while it < max_iterations:
            if time_limit is not None and time.time() - start_time >= time_limit:
                break

            tabu_moves = {move for move, until in tabu_list.items() if until > it}

            if pool is not None:
                tasks = [(current_route, chunk, tabu_moves, best_distance) for chunk in chunks]
                partial_results = pool.starmap(evaluate_moves, tasks)
            else:
                partial_results = [evaluate_moves(current_route, all_moves, tabu_moves, best_distance, cities)]

            best_move, best_move_distance = min(partial_results, key=lambda r: r[1])

            if best_move is not None:
                i, j = best_move
                current_route[i], current_route[j] = current_route[j], current_route[i]
                current_distance = best_move_distance
                tabu_list[best_move] = it + tabu_tenure

                if current_distance < best_distance:
                    best_route = list(current_route)
                    best_distance = current_distance

            results.append(current_distance)
            it += 1
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - start_time
    return best_route, best_distance, results, elapsed


if __name__ == "__main__":
    random.seed(123)
    num_cities = 100
    cities = [(random.randint(0, 1000), random.randint(0, 1000)) for _ in range(num_cities)]

    max_iterations = 200
    tabu_tenure = 25
    num_workers = cpu_count()

    print(f"Liczba miast: {num_cities}, rozmiar sąsiedztwa: {num_cities * (num_cities - 1) // 2}")
    print(f"Liczba iteracji: {max_iterations}, procesy: {num_workers}")

    print("\nTabu Search - sekwencyjnie (1 proces)")
    random.seed(123)
    seq_route, seq_distance, seq_results, seq_time = tabu_search(
        cities, max_iterations=max_iterations, tabu_tenure=tabu_tenure, num_workers=1
    )
    print(f"Całkowity dystans: {seq_distance:.2f}, czas: {seq_time:.2f} s")

    print(f"\nTabu Search - równolegle ({num_workers} procesów)")
    random.seed(123)
    par_route, par_distance, par_results, par_time = tabu_search(
        cities, max_iterations=max_iterations, tabu_tenure=tabu_tenure, num_workers=num_workers
    )
    print(f"Najlepsza trasa: {par_route}")
    print(f"Całkowity dystans: {par_distance:.2f}, czas: {par_time:.2f} s")

    print(f"\nPrzyspieszenie: {seq_time / par_time:.2f}x")

    with open("lista_tabu.txt", "w", encoding="utf-8") as plik:
        for element in par_results:
            plik.write(str(int(element)) + "\n")
