# 🚀 CI2024_lab2

## 🏆 Lab 2: The TSP Problem

This lab required solving the 🗺️ Travelling Salesman Problem (TSP) on five different instances using two algorithms: one fast but approximate ⚡, and one slower but more accurate 🎯. The final code is in `Lab2.ipynb`.

---

### ⚡ Fast Algorithm: Greedy + Simulated Annealing

The fast algorithm consists of a simple greedy heuristic followed by simulated annealing 🔥 to refine the solution. The greedy algorithm constructs a tour by iteratively selecting the nearest unvisited city. Once a valid tour is found, simulated annealing optimizes it using a 2-opt local search.

#### 🏁 Greedy Algorithm
```python
import numpy as np

def greedy(starting_city: int):
    visited = np.full(len(CITIES), False)
    dist = DIST_MATRIX.copy()
    city = starting_city
    visited[city] = True
    tsp = [city]
    while not np.all(visited):
        dist[:, city] = np.inf
        closest = np.argmin(dist[city])
        visited[closest] = True
        city = closest
        tsp.append(city)
    tsp.append(tsp[0])  # Close the cycle
    return tsp
```

#### 🔥 Simulated Annealing
```python
# Simulated annealing parameters
initial_temp = 0.1  
cooling_rate = 0.999  
min_temp = 1e-5  

def two_opt(tour):
    i, j = np.sort(np.random.randint(1, len(CITIES) - 1, size=2))
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

def simulated_annealing(tsp, initial_temp, cooling_rate, min_temp):
    current_solution = tsp.copy()
    current_cost = tsp_cost(current_solution)
    best_solution = current_solution.copy()
    best_cost = current_cost
    temperature = initial_temp

    while temperature > min_temp:
        new_solution = two_opt(current_solution)
        new_cost = tsp_cost(new_solution)
        cost_diff = new_cost - current_cost
        if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temperature):
            current_solution = new_solution
            current_cost = new_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
        temperature *= cooling_rate

    return best_solution, best_cost
```

#### 📊 Results
| 🌍 Instance  | 📏 Cost (km) |
|-------------|------------|
| 🇻🇺 Vanuatu  | 1449.74  |
| 🇮🇹 Italy    | 4274.80  |
| 🇷🇺 Russia   | 42594.00 |
| 🇺🇸 US       | 47836.26 |
| 🇨🇳 China    | 58699.99 |

---

### 🧬 Evolutionary Algorithm (EA)

The second algorithm is a modern Genetic Algorithm 🧪 with a mix of greedy and random individuals in the initial population. The crossover uses the inversion method 🔄, and mutation is handled via 2-opt. Simulated annealing is applied occasionally to offspring for further improvement 📉.

#### 🧩 Evolutionary Algorithm Implementation
```python
POPULATION_SIZE = 150 
GENERATIONS = 300 
ELITE_SIZE = 20  
MUTATION_RATE = 0.5

def evolutionary_algorithm():
    population = create_initial_population(
        POPULATION_SIZE, 0, int(0.95 * POPULATION_SIZE), int(0.05 * POPULATION_SIZE)
    )
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(GENERATIONS):
        population.sort(key=fitness, reverse=True)
        elite = population[:ELITE_SIZE]
        if fitness(elite[0]) > best_fitness:
            best_solution = elite[0]
            best_fitness = fitness(best_solution)
        new_population = elite[:]
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = inversion_crossover(parent1, parent2)
            child = mutate(child, MUTATION_RATE)
            new_population.append(child)
        population = new_population
        if generation % 20 == 0:
            print(f"Generation {generation}: Best tour length = {-best_fitness:.2f} km")
    return best_solution, -best_fitness
```

#### 📊 Results
| 🌍 Instance  | 📏 Cost (km)  | 🔄 Generations |
|-------------|-------------|---------------|
| 🇻🇺 Vanuatu  | 1345.54   | 0           |
| 🇮🇹 Italy    | 4172.76   | 18          |
| 🇷🇺 Russia   | 32984.48  | 300         |
| 🇺🇸 US       | 40728.24  | 300         |
| 🇨🇳 China    | 56416.82  | 300         |

---

## 🔍 Code Review Feedback

### 📌 Suggested Improvements

✅ **Lin-Kernighan Algorithm**: Could improve results over greedy + simulated annealing.
✅ **Multi-Start Greedy**: Running the greedy algorithm multiple times from different starting points.
✅ **Crossover Strategy**: PMX crossover might preserve relative order and position better.
✅ **Population Initialization**: Full randomization might help with solution space exploration.
✅ **Adaptive Mutation Rate**: Increasing the mutation rate when fitness stagnates.

### 🧪 Implemented Tests Based on Review

- **Multi-Start Greedy**: Tried but found it inefficient for larger instances.
- **PMX Crossover**: Implemented but observed premature stagnation.
- **Adaptive Mutation Rate**: Implemented but slowed execution without major fitness gains.

---

## 🎯 Conclusion

The final solution balances **speed** ⚡ and **accuracy** 🎯 using greedy + simulated annealing for fast approximation and an evolutionary algorithm for better optimization. Future improvements could focus on Lin-Kernighan heuristics and dynamic mutation strategies. 🚀
# 🚀 CI2024_lab2

## 🏆 Lab 2: The TSP Problem

This lab required solving the 🗺️ Travelling Salesman Problem (TSP) on five different instances using two algorithms: one fast but approximate ⚡, and one slower but more accurate 🎯. The final code is in `Lab2.ipynb`.

---

### ⚡ Fast Algorithm: Greedy + Simulated Annealing

The fast algorithm consists of a simple greedy heuristic followed by simulated annealing 🔥 to refine the solution. The greedy algorithm constructs a tour by iteratively selecting the nearest unvisited city. Once a valid tour is found, simulated annealing optimizes it using a 2-opt local search.

#### 🏁 Greedy Algorithm
```python
import numpy as np

def greedy(starting_city: int):
    visited = np.full(len(CITIES), False)
    dist = DIST_MATRIX.copy()
    city = starting_city
    visited[city] = True
    tsp = [city]
    while not np.all(visited):
        dist[:, city] = np.inf
        closest = np.argmin(dist[city])
        visited[closest] = True
        city = closest
        tsp.append(city)
    tsp.append(tsp[0])  # Close the cycle
    return tsp
```

#### 🔥 Simulated Annealing
```python
# Simulated annealing parameters
initial_temp = 0.1  
cooling_rate = 0.999  
min_temp = 1e-5  

def two_opt(tour):
    i, j = np.sort(np.random.randint(1, len(CITIES) - 1, size=2))
    return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

def simulated_annealing(tsp, initial_temp, cooling_rate, min_temp):
    current_solution = tsp.copy()
    current_cost = tsp_cost(current_solution)
    best_solution = current_solution.copy()
    best_cost = current_cost
    temperature = initial_temp

    while temperature > min_temp:
        new_solution = two_opt(current_solution)
        new_cost = tsp_cost(new_solution)
        cost_diff = new_cost - current_cost
        if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temperature):
            current_solution = new_solution
            current_cost = new_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
        temperature *= cooling_rate

    return best_solution, best_cost
```

#### 📊 Results
| 🌍 Instance  | 📏 Cost (km) |
|-------------|------------|
| 🇻🇺 Vanuatu  | 1449.74  |
| 🇮🇹 Italy    | 4274.80  |
| 🇷🇺 Russia   | 42594.00 |
| 🇺🇸 US       | 47836.26 |
| 🇨🇳 China    | 58699.99 |

---

### 🧬 Evolutionary Algorithm (EA)

The second algorithm is a modern Genetic Algorithm 🧪 with a mix of greedy and random individuals in the initial population. The crossover uses the inversion method 🔄, and mutation is handled via 2-opt. Simulated annealing is applied occasionally to offspring for further improvement 📉.

#### 🧩 Evolutionary Algorithm Implementation
```python
POPULATION_SIZE = 150 
GENERATIONS = 300 
ELITE_SIZE = 20  
MUTATION_RATE = 0.5

def evolutionary_algorithm():
    population = create_initial_population(
        POPULATION_SIZE, 0, int(0.95 * POPULATION_SIZE), int(0.05 * POPULATION_SIZE)
    )
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(GENERATIONS):
        population.sort(key=fitness, reverse=True)
        elite = population[:ELITE_SIZE]
        if fitness(elite[0]) > best_fitness:
            best_solution = elite[0]
            best_fitness = fitness(best_solution)
        new_population = elite[:]
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = inversion_crossover(parent1, parent2)
            child = mutate(child, MUTATION_RATE)
            new_population.append(child)
        population = new_population
        if generation % 20 == 0:
            print(f"Generation {generation}: Best tour length = {-best_fitness:.2f} km")
    return best_solution, -best_fitness
```

#### 📊 Results
| 🌍 Instance  | 📏 Cost (km)  | 🔄 Generations |
|-------------|-------------|---------------|
| 🇻🇺 Vanuatu  | 1345.54   | 0           |
| 🇮🇹 Italy    | 4172.76   | 18          |
| 🇷🇺 Russia   | 32984.48  | 300         |
| 🇺🇸 US       | 40728.24  | 300         |
| 🇨🇳 China    | 56416.82  | 300         |

---

## 🔍 Code Review Feedback

### 📌 Suggested Improvements

✅ **Lin-Kernighan Algorithm**: Could improve results over greedy + simulated annealing.
✅ **Multi-Start Greedy**: Running the greedy algorithm multiple times from different starting points.
✅ **Crossover Strategy**: PMX crossover might preserve relative order and position better.
✅ **Population Initialization**: Full randomization might help with solution space exploration.
✅ **Adaptive Mutation Rate**: Increasing the mutation rate when fitness stagnates.

### 🧪 Implemented Tests Based on Review

- **Multi-Start Greedy**: Tried but found it inefficient for larger instances.
- **PMX Crossover**: Implemented but observed premature stagnation.
- **Adaptive Mutation Rate**: Implemented but slowed execution without major fitness gains.

---

## 🎯 Conclusion

The final solution balances **speed** ⚡ and **accuracy** 🎯 using greedy + simulated annealing for fast approximation and an evolutionary algorithm for better optimization. Future improvements could focus on Lin-Kernighan heuristics and dynamic mutation strategies. 🚀
