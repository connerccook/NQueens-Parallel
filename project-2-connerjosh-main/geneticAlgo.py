from random import randint
from time import time
import random
import sys

from mpi4py import MPI #MPI library


# Performance Decorator to track how fast our program is
def performance(fn):
  def wrapper(*args, **kwargs):
    t1 = time()
    result = fn(*args, **kwargs)
    t2 = time()
    print(f'This function took {t2-t1} s to run')
    return result
  return wrapper

@performance
def genetic_search(problem=0, ngen=5000, pmut=0.1, n=100):
    """Call genetic_algorithm on the appropriate parts of a problem.
    This requires the problem to have states that can mate and mutate,
    plus a value method that scores states."""
    comm = MPI.COMM_WORLD
    size = comm.Get_size() #Number of processes
    rank = comm.Get_rank()
    CHILD_ITERATIONS = 5
    res = None
    solution = None
    
    #Parent receives problem parameters, creates gene pool and broadcasts to all chidlren
    gene_pool = list(range(problem)) if rank == 0 else None
    
    #Children receive gene pool
    gene_pool = comm.bcast(gene_pool, root = 0)
    NSize = len(gene_pool)  #Size of chess board is derived from gene pool
    
    #Every process generates a subpopulation
    sub_pop_size = int(n / size)
    if rank == 0:
        sub_pop_size = n - (size-1)*sub_pop_size
    sub_population = init_population(sub_pop_size, gene_pool)
    #Parent process gathers subpopulations to create full population
    grid_population = comm.gather(sub_population, root=0)
    
    #Parent process gathers subpopulations to create full population
    if rank==0:
        #Merge subpopulations
        population = mergeLists(grid_population)
        #Begin listening for solution
        solution_req = comm.irecv(res)

    while True:
        #[===Scatter population to child processes===]
        if rank == 0:
            random.shuffle(population)
            population = splitList(population, size)
        else:
            population = None
    
        sub_population = comm.scatter(population, root=0)
        sub_pop_size = len(sub_population)
        
        
        #[===Select, cross===]
        for _child_iter in range(CHILD_ITERATIONS):
            #3-way tournament selection to combine selections and crossover
            res = select_3way_tourny(sub_population, sub_pop_size, gene_pool)
            if res:
                comm.isend(res, 0, 0)
        
        #Mutate 
        for i in range(len(sub_population)):
            sub_population[i] = mutate(sub_population[i], pmut)
        
        #[===Recombine population for shuffling===]
        grid_population = comm.gather(sub_population, root=0)
        if rank == 0:
            population = mergeLists(grid_population)
            
            #Check if a solution was found
            hasSolution, solution = solution_req.test()
        #If solution found broadcast to all processes and stop genetic algo
        solution = comm.bcast(solution, root = 0)
        if(solution):
            return solution

# split the population into subpopulations for each child
def splitList(list, n):
    newList = []
    i0 = 0
    increment = int(len(list)/n)
    if increment*n < len(list):
        increment+=1
    
    for i1 in range(increment, len(list), increment):
        newList.append( list[i0:i1] )
        i0 = i1
    if i1 < len(list):
        newList.append( list[i1:] )
    return newList

# Merge the subpopulations into one population
def mergeLists(grid_list):
    list = []
    for sublist in grid_list:
        list.extend(sublist)
    return list

# inilize the population
# creates each chromosome 
def init_population(pop_number, gene_pool):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals
    """
    population = []
    for i in range(pop_number):
        #Create new chess board config with no row and column conflicts
        new_individual = gene_pool.copy();
        random.shuffle(new_individual);
        
        population.append(new_individual)

    return population

def pmx_crossover(dad, mom, gene_pool):
    _gene_pool = gene_pool.copy()
    random.shuffle(_gene_pool)
    child = []
    for i in range(len(dad)):
        if dad[i] == mom[i]:
            child.append( dad[i] )
        else:
            while True:
                gene = _gene_pool.pop()
                if gene not in child:
                    child.append( gene )
                    break
    return child
            

def select_3way_tourny(pop, sub_pop_size, gene_pool):
    #Select 3 individuals from subpopulation at random
    candidates = []
    weakestCandidate = -1
    weakestFitness = sys.maxsize
    while len(candidates) < 3:
        tmpCandidate = random.randrange(0, sub_pop_size)
        if tmpCandidate not in candidates:
            candidates.append(tmpCandidate)
            candidateFitness = fitness_fn_naive_conflict(pop[tmpCandidate])
            if candidateFitness == 0:
                #If solution found, return it
                return pop[tmpCandidate]
            if candidateFitness < weakestFitness:
                weakestFitness = candidateFitness
                weakestCandidate = tmpCandidate
    
    #Replace weakest candidate with child of two stronger candidates
    dad = pop[candidates[0]] if candidates[0] != weakestCandidate else pop[candidates[2]]
    mom = pop[candidates[1]] if candidates[1] != weakestCandidate else pop[candidates[2]]
    pop[weakestCandidate] = pmx_crossover(dad, mom, gene_pool)
    return None

def mutate(individual, pmut):
    '''Mutate by swapping two positions to maintain validity'''
    if random.uniform(0, 1) <= pmut:
        n = len(individual)
        #Pick 2 random points
        m1 = random.randrange(0, n)
        m2 = random.randrange(0, n)
        #Swap points
        n = individual[m1]
        individual[m1] = individual[m2]
        individual[m2] = n
    return individual

# calculate the fitness score
def fitness_fn_naive_conflict(q):
    num_conflicts = 0
    size = len(q)
    for (r1, c1) in enumerate(q):
        for (r2, c2) in enumerate(q):
            if (r1, c1) != (r2, c2):
                num_conflicts += 1 if (r1 - c1 == r2 - c2 or r1 + c1 == r2 + c2) else 0
    return num_conflicts
    
def fitness_fn_conf(q):
    n = len(q)
    left_diagonal = [0]*(2*n-1)
    right_diagonal = left_diagonal.copy()
    
    for i in range(1, n):
        left_diagonal[i+q[i]]+=1
        right_diagonal[n-i+q[i]]+=1
    sum = 0
    for i in range(1, 2*n-1):
        counter = 0
        if left_diagonal[i] > 1:
            counter += left_diagonal[i]-1
        if right_diagonal[i] > 1:
            counter += right_diagonal[i]-2
        sum += counter / (n-abs(i-n))
    return sum