import argparse
import os
from queue import PriorityQueue
import queue

#definicija funkcija za čitanje podataka
def read_file(path_to_states):
    start=""
    goal=""
    f = open(path_to_states, 'r', encoding='utf-8')
    lines = f.readlines()
    states={}
    for line in lines:
        if line.startswith("#"):
            continue
        spliited = line.strip().split(" ")
        if(start==""):
            start=spliited[0]
            continue
        if(goal==""):
            if(len(spliited)>1):
                goal=spliited
            else:
                goal=spliited[0]
            continue
        node=spliited[0].strip(":")
        others = spliited[1:]
        neighbors={}
        for one in others:
            neighbor = one.split(",")[0]
            cost = one.split(",")[1]
            neighbors[neighbor] = float(cost)
        states[node]=neighbors
    f.close()
    return states, start, goal

#defincija funkcije za čitanje heuristike
def read_heuristic(path_to_heuristic):
    f = open(path_to_heuristic, 'r', encoding='utf-8')
    lines = f.readlines()
    heuristic = {}
    for line in lines:
        node = line.split(" ")[0].strip(":")
        heuristic_value = line.split(" ")[1]
        heuristic[node]=float(heuristic_value)
    f.close()
    return heuristic

#algoritam BFS ostvaren je strukturom red gdje je svaki čvor lista koja se sastoji od cijene, imena čvora i puta do tog čvora
#također koristimo set koji se zove closed gdje spremamo čvorove koje smo već prošli kako ne bi morali ponavljati postupak
#funkcija vraća put do ciljnog čvora, cijenu puta te duljinu liste zatvorenih čvorova
def bfs(states, start_state, goal):
    open = ([(0.0, start_state, [start_state])])
    closed = set()
    while len(open)>0:
        node = open.pop(0)
        unvisited = set(states.keys()) - closed
        if node[1] in unvisited:
            closed.add(node[1])
            if node[1] in goal:
                return node[2], node[0], len(closed)
            sorted_next = sorted(states[node[1]].items())
            for element in sorted_next:
                next = element[0]
                next_cost = element[1]
                if next not in closed:
                    new_cost = node[0] + next_cost
                    new_path = node[2] + [next]
                    open.append((new_cost, next, new_path))
    return [],[],[]

#algoritam UCS ostvaren je pomoću prioritetnog reda u koji spremamo čvorove u odnosu na to kolika im je cijena
#sve ostalo je identično kao i kod algoritma BFS
def ucs(states, start_state, goal):
    open = PriorityQueue()
    open.put((0, start_state, [start_state]))
    closed = set()
    while not open.empty():
        node = open.get()
        unvisited = set(states.keys()) - closed
        if node[1] in unvisited:
            closed.add(node[1])
            if node[1] in goal:
                return node[2], node[0], len(closed)
            for element in states[node[1]].items():
                next=element[0]
                next_cost =element[1]
                if next not in closed:
                    open.put((node[0] + next_cost, next, node[2] + [next]))
    return [],[],[]

#astar algoritam je malo kompliciraniji i za njega smo trebali koristit i heurističku funkciju
#ostvaren je preko prioritenog reda u kojeg spremamo liste s obzirom na prvi element te liste-f(n) što je zbroj cijene do čvora n (g(n)) te heuristike tog čvora n
def astar(states, heuristic, start_state, goal):
    open = PriorityQueue()
    open.put((float(heuristic[start_state]), start_state, [start_state]))
    closed = set()
    node_cost = {start_state: 0}    
    while not open.empty():
        node = open.get()
        closed.add(node[1])
        if node[1] in goal:
            return node[2], node_cost[node[1]], len(closed)
        for element in states[node[1]].items():
            next = element[0]
            next_cost = element[1]
            unvisited = set(states.keys()) - closed
            if next in unvisited:
                g_value = node_cost[node[1]] + next_cost
                if next not in node_cost or g_value < node_cost[next]:
                    node_cost[next] = g_value
                    f_value = g_value + heuristic[next]
                    new_path = node[2] + [next]
                    open.put((f_value, next, new_path))                               
    return [],[],[]

#u ovoj funkciji provjeravamo je li neka heuristika konzistenta, a to znači da mora ispuniti uvjet da za svakog susjeda n čvora s vrijedi da h(s) <= h(n) + c, gdje je c cijena između čvora i susjeda
def check_consistent(states, heuristic):
    isConsistent = True
    for state in states.keys():
        for neighbor in states[state]:
            if(float(heuristic[state]) <= float(heuristic[neighbor]) + float(states[state][neighbor])):
                print("[CONDITION]: [OK] h({state}) <= h({neighbor}) + c: {hs1} <= {hs2} + {c}".format(state=state, neighbor=neighbor, hs1=str(float(heuristic[state])), hs2=str(float(heuristic[neighbor])), c=str(float(states[state][neighbor]))))
            else:
                isConsistent= False
                print("[CONDITION]: [ERR] h({state}) <= h({neighbor}) + c: {hs1} <= {hs2} + {c}".format(state=state, neighbor=neighbor, hs1=str(float(heuristic[state])), hs2=str(float(heuristic[neighbor])), c=str(float(states[state][neighbor]))))

    print("[CONCLUSION]: Heuristic is consistent." if isConsistent else "[CONCLUSION]: Heuristic is not consistent.")

#u ovoj funkciji provjeravamo je li neka heuristika optimisična, a to znači da za svaki čvor s vrijedi da je njegova heuristika h(s) <= h*, gdje je h* cijena puta od stanja s do ciljnog stanja
#h* smo dobili koristeći algoritam UCS koji smo gore opisali zato jer je taj algoritam potpun i optimalan i uvijek ćemo biti sigurni da smo dobili najmanju moguću cijenu, dok to nije sigurno kod BFS-a
def check_optimistic(states, heuristic):
    isOptimistic = True
    for state in states.keys():
        _, h_star, _ = ucs(states, state, goal)
        if(float(heuristic[state]) <= float(h_star)):
            print("[CONDITION]: [OK] h({state}) <= h*: {h_state} <= {h_star}".format(state=state, h_state=str(float(heuristic[state])), h_star=str(float(h_star))))
        else:
            isOptimistic = False
            print("[CONDITION]: [ERR] h({state}) <= h*: {h_state} <= {h_star}".format(state=state, h_state=str(float(heuristic[state])), h_star=str(float(h_star))))

    print("[CONCLUSION]: Heuristic is optimistic." if isOptimistic else "[CONCLUSION]: Heuristic is not optimistic.")


#ovdje slijedi glavni program, učitavanje podataka pomoću glavne linije te pozivanje funkcija s obzirom na to što je korisnik upisao 
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str)
parser.add_argument('--ss', type=str)
parser.add_argument('--h', type=str)
parser.add_argument('--check-optimistic', action='store_true', dest='check_optimistic')
parser.add_argument('--check-consistent', action='store_true', dest='check_consistent')
args = parser.parse_args()

algorithm = args.alg
path_to_states = args.ss
path_to_heuristic = args.h

if(algorithm is not None):
    if(algorithm.upper()=="BFS"):
        print("# BFS")
        states, start, goal = read_file(path_to_states)
        path_to_goal, cost_to_goal, states_visited = bfs(states, start, goal)
        if not path_to_goal:
            print("[FOUND_SOLUTION]:")
        else:
            print("[FOUND_SOLUTION]: yes")
            print("[STATES_VISITED]: %d" %states_visited)
            print("[PATH_LENGTH]: %d" %len(path_to_goal))
            print("[TOTAL_COST]: %.1f" %float(cost_to_goal))
            print("[PATH]: %s" %" => ".join(path_to_goal))    
    elif(algorithm.upper()=="UCS"):
        print("# UCS")
        states, start, goal = read_file(path_to_states)
        if(ucs(states, start, goal)==[]):
            print("[FOUND_SOLUTION]:")           
        else:
            path_to_goal, cost_to_goal, states_visited = ucs(states, start, goal)
            print("[FOUND_SOLUTION]: yes")
            print("[STATES_VISITED]: %d" %states_visited)
            print("[PATH_LENGTH]: %d" %len(path_to_goal))
            print("[TOTAL_COST]: %.1f" %float(cost_to_goal))
            print("[PATH]: %s" %" => ".join(path_to_goal))
    elif(algorithm.upper()=="ASTAR"):
        print("# ASTAR %s" %os.path.basename(path_to_heuristic))
        states, start, goal = read_file(path_to_states)
        heuristic = read_heuristic(path_to_heuristic)
        path_to_goal, cost_to_goal, states_visited = astar(states, heuristic, start, goal)
        if path_to_goal==[]:
            print("[FOUND_SOLUTION]:")
        else:
            print("[FOUND_SOLUTION]: yes")
            print("[STATES_VISITED]: %d" %states_visited)
            print("[PATH_LENGTH]: %d" %len(path_to_goal))
            print("[TOTAL_COST]: %.1f" %float(cost_to_goal))
            print("[PATH]: %s" %" => ".join(path_to_goal))
elif(algorithm is None):
    if(path_to_states is not None and path_to_heuristic is not None):
        if(args.check_optimistic):
            print("# HEURISTIC-OPTIMISTIC %s" %os.path.basename(path_to_heuristic))
            states, start, goal = read_file(path_to_states)
            heuristic = read_heuristic(path_to_heuristic)
            check_optimistic(states, heuristic)
        if(args.check_consistent):
            print("# HEURISTIC-CONSISTENT %s" %os.path.basename(path_to_heuristic))
            states, start, goal = read_file(path_to_states)
            heuristic = read_heuristic(path_to_heuristic)
            check_consistent(states, heuristic)
