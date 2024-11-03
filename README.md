# CI2024_lab2
## Lab 2: the TSP problem

The text for lab 2 required to solve the TSP problem (on 5 different instances) using two algorithms: one fast but approximate and a slower, yet more accurate. Moreover, it was requested to report the final cost and the number of steps (since it's not specified what is the fitness in this context, I counted as 'steps' the number of generations of the EA used as second algorithm). The final version of the code delivered is in the file Lab2.ipynb.  
For what concerns the fast algorithm, the solution provided is quite simple. First of all, a very simple but effective greedy algorithm is used to find a path from an initial random city. The following step in the path is simply chosen as the closest city from the current one. Then, when the algorithm returns a valid solution, simulated annealing is applied in order to try to reduce the cost. Parameters used in the simulated annealing have been fine-tuned following an 'empirical' strategy: trial and errors until I found some values that seemed to give reasonable results in the 5 instances. Moreover, the tweak function used in the simulated annealing is 2-opt. Also in this case, I tried many different strategies, such as 3-opt, 4-opt and so on... I report the code:
```python
def greedy(startingcity: int):
    visited = np.full(len(CITIES), False)
    dist = DIST_MATRIX.copy()
    city = startingcity
    visited[city] = True
    tsp = list()
    tsp.append(int(city))
    while not np.all(visited):
        dist[:, city] = np.inf
        closest = np.argmin(dist[city])
        visited[closest] = True
        city = closest
        tsp.append(int(city))
    tsp.append(tsp[0])
    return tsp

#Choose a random city and execute the algorithm
city = np.random.randint(len(CITIES))
tsp = greedy(city)
```
And here we have the simulated annealing algorithm: 
```python
# Simulated annealing parameters 
initial_temp = 0.1  # Starting temperature
cooling_rate = 0.999  # How to slow down the cooling
min_temp = 1e-5  # Minimum temperature 

def two_opt(tour):
    """2-opt to invert a portion of the tour"""
    i, j = np.sort(np.random.randint(1, len(CITIES) - 1, size=2))
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

def simulated_annealing(tsp, initial_temp, cooling_rate, min_temp):
    current_solution = tsp.copy()
    current_cost = tsp_cost(current_solution)
    best_solution = current_solution.copy()
    best_cost = current_cost
    temperature = initial_temp

    while temperature > min_temp:
        # Generate a new solution through 2-opt
        new_solution = two_opt(current_solution)

        # Cost of the new solution
        new_cost = tsp_cost(new_solution)

        # Cost difference
        cost_diff = new_cost - current_cost

        # Decide whether to accept the new solution
        if cost_diff < 0 or np.random.rand() < np.exp(-cost_diff / temperature):
            current_solution = new_solution
            current_cost = new_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

        # Cool down the temperature
        temperature *= cooling_rate

    return best_solution, best_cost

best_solution, best_cost = simulated_annealing(tsp, initial_temp, cooling_rate, min_temp)
```
This algorithm has proved to be really fast on all the instances and it is able to provide some good results (anyway, they are far from the optimum for the larger collections of cities).
I report here the results obtained for Vanuatu, Italy, Russia,US and China:


|   Instance   |    Cost     |
|--------------|-------------|
| Vanuatu      | 1449.74  km |
| Italy        | 4274.80  km |
| Russia       | 42594.00 km |
| US           | 47836.26 km |
| China        | 58699.99 km |  



I didn't know in this case how to count the number of steps.
For what concerns the second algorithm, the strategy used is much more complex and time consuming. I have done a lot of trials when realizing the EA. I try to report here the flow of tests performed and the different types of algorithms that I have tried. First of all, when first tried to realize it, I had to decide between the three common flows of execution in Genetic Algorithms: Historical, Modern and Hyper-modern. I decided to try all of them but, after some trials, I have noticed that the one that had better results was the modern approach. In the file trials_tsp.ipynb are left the three algorithms and their implementations. Last trials for parameters tuning have been carried only on the Modern flow. In the code I will paste here, there is the reference to a fitness function that, given the type of problem, I simply defined as -tsp_cost (that is the cost for travelling all the edges of the cycle). I report here the code of the evolutionary algorithm: 
```python
# Algorithm parameters
POPULATION_SIZE = 150 
GENERATIONS = 300 
ELITE_SIZE = 20  
MUTATION_RATE = 0.5

# EA -> Modern GA
def evolutionary_algorithm():
    # Initial population setup
    population = create_initial_population(
        POPULATION_SIZE, int(POPULATION_SIZE * 0), int(0.95 * POPULATION_SIZE), int(0.05 * POPULATION_SIZE)
    )
    best_solution = None
    best_fitness = float('-inf')

    for generation in range(GENERATIONS):
        # Sort the population by fitness in descending order
        population.sort(key=fitness, reverse=True)
        
        # Select the best individual as elite
        elite = population[:ELITE_SIZE]  # Only keep the top individual
        
        # Update the best individual if needed
        if fitness(elite[0]) > best_fitness:
            best_solution = elite[0]
            best_fitness = fitness(best_solution)

        # New population setup
        new_population = elite[:]  # Keep only the best individual
        while len(new_population) < POPULATION_SIZE:
            # Selection and crossover
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = inversion_crossover(parent1, parent2)
            child = mutate(child, MUTATION_RATE)
            new_population.append(child)
        
        # Update population for the next generation
        population = new_population

        # Display progress every 20 generations
        if generation % 20 == 0:
            print(f"Generation {generation}: Best tour length = {-best_fitness:.2f} km")

    return best_solution, -best_fitness

# Run the evolutionary algorithm
best_tour, best_cost = evolutionary_algorithm()

print(f"Best tour found: {best_tour} with total length {best_cost:.2f} km")
```
As we can observe in the code, there is the call to many functions that will be better explained in the following. At the end of the execution, simulated annealing is again applied on the best tour found by the evolutionary algorithm. This improvement has demonstrated to work and to reduce the fitness in some cases, while it is not time consuming, so, why not?
Another very important point of the EA algorithm is the initial population, set, in the previous code, by means of the 'create_initial_population' function. At the beginning, I left it completely randomic and I noticed that I wasn't getting many improvements. The fitness was flat after a certain point on so, after some trials, I understood the main problem were the individuals in the initial population. Then, when I tried to change it, I was able to experience more consistent results with those that I was expecting. I tried three different types of starting point at the beginning: randomic, greedy and greedy + simulated annealing (that is actually the first algorithm delivered). Then, the idea was to try to mix them in a certain percentage in order to give the algorithm some fit individuals but not enough to get it stuck in a local optima and still have the possibility to explore the solution space. As I did before, I tried many values and different combinations. For this purpose, I realized a function able to return an initial population with different percentages of greedy individuals, randomic and greedy + simulated annealing. Here is the code:
```python
def create_initial_population(total_size,greedy_size,random_size,simulated_annealing_size):
    population = []
    for _ in range(greedy_size):
        population.append(greedy(np.random.randint(0,len(CITIES))))
    for _ in range(random_size):
        population.append(create_random_solution())
    for _ in range(simulated_annealing_size):
        population.append(simulated_annealing(greedy(np.random.randint(0,len(CITIES))),initial_temp,cooling_rate,min_temp)[0])
    for _ in range(total_size-len(population)):
        population.append(create_random_solution())
    return population
```
At the end, after many trials, the configuration that gave me the best results was: 95% of the population composed of random individuals and, the remaining 5%, fit individuals generated with greedy + simulated annealing.  
There are two other important details that must be defined when designing an EA: the mutation and the crossover function. For what concerns the mutation, driven by the results obtained in previous trials with the simulated annealing, I decided to use 2-opt, while, for the crossover function, I used the inver Over strategy. In later trials I also introduced the scramble mutation. At the end I dedcided to leave both of them in the code. I do not see so many differences in the results, in particular because there is some randomness by definition in the way the evolution algorithm proceeds. As a results, some trials highlights better performances for the 2-opt, some others do the opposite and would made me choose the scramble mutation. If you try the code, both of them are working, you can select which one execute by simply changing one line of code regarding the type of mutation called.   
Here is the code:
```python
def inversion_crossover(parent1, parent2):
    # Determine the size of the parents (excluding the last element)
    size = len(parent1) - 1  
    # Randomly choose two crossover points
    start, end = sorted(random.sample(range(size), 2))

    # Copy the segment from parent1 and invert it
    child_segment = parent1[start:end + 1][::-1]  # Invert the selected segment
    
    # Create a child list that initially includes None
    child = [None] * size
    
    # Fill in the inverted segment into the child
    child[start:end + 1] = child_segment
    
    # Fill the remaining positions with genes from parent2, avoiding duplicates
    p2_index = 0
    for i in range(size):
        if child[i] is None:
            # Find the next gene from parent2 that isn't already in the child
            while parent2[p2_index] in child:
                p2_index += 1
            child[i] = parent2[p2_index]

    child.append(child[0])  # Close the cycle by adding the first city again
    if(random.random()<0.05): #first tests were done with 0.01
        child,_=simulated_annealing(child,initial_temp,cooling_rate,min_temp)
    return child

#2-opt
def mutate(solution, mutation_rate):
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(1, len(solution) - 1), 2))
        solution[i:j] = reversed(solution[i:j])
    return solution

def scramble_mutation(solution, mutation_rate):
    # Apply mutation based on mutation rate
    if random.random() < mutation_rate:
        # Randomly select two indices to define the segment
        i, j = sorted(random.sample(range(1, len(solution) - 1), 2))  # Exclude the first and last cities to keep the tour valid
        # Scramble the segment between the two indices
        scrambled_segment = solution[i:j]
        random.shuffle(scrambled_segment)
        # Replace the segment in the solution with the scrambled segment
        solution[i:j] = scrambled_segment
    return solution
```
For what concerns the crossover function, I tried many other possibilities: order, partially mapped, cycle and uniform crossover. The code regarding this (less successful) trials is left to 'trials_tsp.ipynb' again. As you may have noticed from the code above, there is another important difference with respect to the traditional inversion crossover: simulated annealing is applied on the child generated from the two selected parents with a (low=5%) probability. This was the change in the code that, towards the end, allowed me to obtain the best results among all trials. The problem related to this strategy is that it slows down a lot the execution time on my computer, that's the reason why I kept a low number of generations, even if I know that the algorithm could benefit from more iterations.  
The last detail regards parent selection for the offspring: I decided to use tournament selection as suggested by the professor. Also in this case, the parameter (tau) has been fine-tuned based on results observed on different executions. At the end, I kept it equal to 30. Here is the code:
```python
# Tournament selection
def tournament_selection(population, k=30):
    selected = random.sample(population, k)
    selected.sort(key=fitness, reverse=True)
    return selected[0]
```
 Also in this case, as done with all the other numerical parameters, many possible configurations have been tried. Almost all the trials I have done are tracked in the aforementioned file, in order to help myself remembering which ones were working better. Also in this case, I report a table with costs obtained when executing the second algorithm (I avoid reporting the number of generations, since, as it is clear from the code, it is fixed to 150). I have tried to make some parameters adaptive, such as the mutation rate or the number of generations, but this strategy didn't give the results I was hoping for. For example, for what concerns the mutation rate, I tried to change it based on the standard deviation of the fitness inside the population, increasing or decreasing it. Anyway, as I imagined when I designed it, it slowed down the execution without being worth it from the point of view of cost. So, at the end, I didn't insert it in the delivered code.   
|   Instance   |    Cost     |
|--------------|-------------|
| Vanuatu      | 1345.54  km |
| Italy        | 4172.76  km |
| Russia       | 32984.48 km |
| US           | 40728.24 km |
| China        | 56416.82 km |  


As it can be seen, it seems it's not working very well for China. I tried to increase the number of generations for that particular instance to 700, it takes much more time obviously, but it decreases to 56416.82 km and it was still improving. I don't know what is the optimum in this case because Wolfram is not able to compute it for such a big instace. Additional note: I could have decreased the number of iterations for smaller instances because I actually know what is the optimum and it is reached in a few number of generations (typically not more than 40). Anyway, I think this is not a general strategy: the algorithm should be done in order to work with whatever instance passed as an input, if the format is choerent. Thus, I decided not to introduce a sort of adaptive stopping criteria. Regarding what I have been able to test, 300 generations makes the algorithm in general slow (it takes time), but it's able to obtain good results, in particular if compared to the optimum that we had available.



