from mpi4py import MPI
import time
import random

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
indent = "    " * rank

left = None
right = None

if rank == 0:
    left = "dirty"
    right = "dirty"
elif rank > 0 and rank != size - 1:
    left = None
    right = "dirty"

if size !=2:
    left_neighbour = (rank-1) if rank !=0 else (size-1)
    right_neighbour = (rank+1) if rank !=(size-1) else 0
else:
    if rank==0:
        left_neighbour=1
        right_neighbour=1
    elif rank==1:
        left_neighbour=0
        right_neighbour=0

requests = []

def isAnythingSent():
    return comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

def cleanForkAndSend(neighbour, isLeftFork):
    comm.send((rank, 1, isLeftFork), dest=neighbour)
    print("{}Ja sam {} i saljem svoju vilicu filozofu {}".format(indent, rank, neighbour), flush=True)

def sendRequest(neighbour, isLeftFork):
    comm.send((rank, 0, isLeftFork), dest=neighbour)
    print("{}Ja sam {} i treba mi lijeva vilicu. Pitat cu {}".format(indent, rank, neighbour), flush=True) if isLeftFork else print("{}Ja sam {} i treba mi desna vilica. Pitat cu {}".format(indent, rank, neighbour), flush=True)

def think_and_answer():
    global left, right
    print("{}Ja sam {} i mislim.".format(indent, rank), flush=True)
    sleep_time = random.uniform(1, 5)
    check_request_time = sleep_time/5
    for _ in range(int(sleep_time)):
        if isAnythingSent():
            message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            if message[1]==0:
                if message[0] == left_neighbour and left == "dirty":
                    left=None
                    cleanForkAndSend(message[0], True)
                elif message[0] == right_neighbour and right == "dirty":
                    right=None
                    cleanForkAndSend(message[0], False)
        time.sleep(check_request_time)

def eat():
    global left, right
    eating_time = random.uniform(1, 5)
    print("{}Ja sam {} i jedem.".format(indent, rank), flush=True)
    left = "dirty"
    right = "dirty"
    time.sleep(eating_time)

while True:

    think_and_answer()
    print("{}Ja sam {} i gladan sam!".format(indent, rank), flush=True)

    while (left is None or right is None):
        if left is None:
            sendRequest(left_neighbour, True)
        elif right is None:
            sendRequest(right_neighbour, False)
        while True:
            if isAnythingSent():    
                message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if message[1]==0:
                    if message[0]==left_neighbour and left =="dirty" and not message[2]:
                        left=None
                        cleanForkAndSend(message[0], True)
                    elif message[0] == right_neighbour and right == "dirty" and message[2]:
                        right=None
                        cleanForkAndSend(message[0], False)
                    else:
                        requests.append(message)
                else:
                    if message[0]==left_neighbour and not message[2]:
                        left = "clean"
                        print("{}Ja sam {} i dobio sam svoju lijevu vilicu".format(indent, rank))
                        break
                    elif message[0]==right_neighbour and message[2]:
                        right = "clean"
                        print("{}Ja sam {} i dobio sam svoju desnu vilicu".format(indent, rank))
                        break


    eat()
    if len(requests) > 0:
        for request in requests:
            if request[0]==left_neighbour and left=="dirty":
                left=None
                cleanForkAndSend(request[0], True)
            elif request[0]==right_neighbour and right=="dirty":
                right=None
                cleanForkAndSend(request[0], False)
    requests.clear()
                
    


    

    

    
    