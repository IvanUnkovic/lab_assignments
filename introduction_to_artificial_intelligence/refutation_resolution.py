import sys
import re

goal_line=""

def read_file(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    count=0
    SoS=set()
    premises={}
    for line in lines:
        count+=1
        if line.startswith("#"):
            continue
        if (count == len(lines)):
            line = line.lower()
            goal_line=line
            line = re.sub(r'(?<=\s)v(?=\s)', ':', line)
            line = line.replace("\n", "").replace(" ", "")
            line_array = line.split(":")
            for i in range(len(line_array)):
                if line_array[i].startswith("~"):
                    line_array[i] = line_array[i].strip("~")
                else:
                    line_array[i] = "~" + line_array[i]
            for formula in line_array:
                premises[len(premises)+1]=[formula]
                SoS.add(len(premises))
        else:
            line = line.lower()
            line = re.sub(r'(?<=\s)v(?=\s)', ':', line)
            line = line.replace("\n", "").replace(" ", "")
            formulas=line.split(":")
            premises[len(premises)+1]=formulas
    return premises, goal_line, SoS

def read_cooking_file(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    premises={}
    for line in lines:
        if line.startswith("#"):
            continue
        line = line.lower()
        line = re.sub(r'(?<=\s)v(?=\s)', ':', line)
        line = line.replace("\n", "").replace(" ", "")
        formulas=line.split(":")
        premises[len(premises)+1]=formulas
    return premises

def read_instructions(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    lines = f.readlines()
    instructions=[]
    for line in lines:
        if line.startswith("#"):
            continue
        if(len(line.split(" "))>2):
            line = line.lower()
            line = line.replace("\n", "")
            parts = line.rsplit(" ",1)
            instructions.append(parts)
        else:
            line=line.lower()
            line=line.replace("\n","")
            instruction=line.split(" ")
            instructions.append(instruction)    
    return instructions

def isTautology(clause):
    positive=set()
    negative=set()
    for literal in clause:
        if(literal[0]=="~"):
            negative.add(literal.strip("~"))
        else:
            positive.add(literal)
    if len(positive.intersection(negative)) > 0:
        return True
    else:
        return False
    
def checkNIL(premises, old):
    for old1 in old:
        for old2 in old:
            if(len(premises[old1])==1):
                if(len(premises[old2])==1):
                    first=list(premises[old1])[0]
                    second=list(premises[old2])[0]
                    if (first[0]=="~" and second == first.strip("~")) or (second[0]=="~" and first == second.strip("~")):
                        return True

def findifNIL(premises, SoS, old, found):
    new={}
    for numberSoS in SoS:
        if numberSoS in premises.keys():
            for literalSoS in premises[numberSoS]:
                want_to_find=""
                if (literalSoS.startswith("~")):
                    want_to_find=literalSoS.strip("~")
                else:
                    want_to_find="~"+literalSoS
                for numberOld in old:
                    for literalOld in premises[numberOld]:
                        if(literalOld==want_to_find):
                            new_clause = frozenset(premises[numberOld].union(premises[numberSoS]) - {literalOld, literalSoS})
                            new[new_clause]=([numberOld, numberSoS])    
    if(len(new)>0):
        for new_clause, numbers in new.items():
            #moramo provjeriti je li redundantna ili tautologija
            if isTautology(new_clause):
                continue
            removed=set()
            for i in range(len(premises)):
                    if(new_clause.issubset(premises[i+1])):
                        removed.add(i+1)
            new_premises = premises.copy()
            new_premises[len(premises)+1]=new_clause
            new_old = old
            new_old.add(numbers[1])
            new_old = {x for x in new_old if x not in removed}
            new_SoS = SoS
            if numbers[1] in new_SoS:
                new_SoS.remove(numbers[1])
            new_SoS.add(len(new_premises))
            if(len(new_clause)==0):
                if not found:
                    if(checkNIL(new_premises, old)):
                        found=True
                        path.append([list(new_clause), numbers])
                        return True
            if(len(new_clause)>0):
                answer = findifNIL(new_premises, new_SoS, new_old, found)
                if answer:
                    path.append([list(new_clause), numbers])
                    return True
    else:
        return False
    return False

if(len(sys.argv)==3 or len(sys.argv)==4):
    if(sys.argv[1]=="resolution"):
        found = False
        path=[]
        premises, goal_line, SoS = read_file(sys.argv[2])
        lenght_premises = len(premises)
        for key in premises:
            premises[key] = set(premises[key])
        useable_clauses_numbers = list(premises.keys())
        #prvo micemo nadskupe
        removed=set()
        for i in range(len(premises)):
            for j in range(len(premises)):
                if (i!=j):
                    if(premises[i+1].issubset(premises[j+1])):
                        removed.add(j+1)
        useable_clauses_numbers = {x for x in useable_clauses_numbers if x not in removed}
        for i in useable_clauses_numbers:
            if(isTautology(premises[i])):
                useable_clauses_numbers.remove(i)
        old = set(useable_clauses_numbers)
        for i in SoS:
            if(i in old):
                old.remove(i)
        length = len(premises)
        isDeductive = findifNIL(premises, SoS, old, found)    
        if(isDeductive):
            path=path[::-1]
            path_numbers=set()
            for element in path:
                path_numbers.add(element[1][0])
                path_numbers.add(element[1][1])
            for number in path_numbers:
                if number<=lenght_premises:
                    print(str(number) + ". " + " v ".join(premises[number]))
            print("=" * 15)
            counter=length
            for element in path:
                counter+=1
                if not element[0]:
                    print(str(counter) + ". NIL " + "(" + str(element[1][0]) + ", " + str(element[1][1]) + ")")
                else:
                    print(str(counter) + ". " + " v ".join(element[0]) + " " + "(" + str(element[1][0]) + ", " + str(element[1][1]) + ")")
            print("=" * 15)  
            print("[CONCLUSION]: {} is true".format(goal_line.strip("\n")))
        else:
            print("=" * 15)
            print("[CONCLUSION]: {} is unknown".format(goal_line.strip("\n")))

    elif(sys.argv[1]=="cooking"):
        path=[]
        found=False
        premises=read_cooking_file(sys.argv[2])
        print("Constructed with knowledge:")
        for key, value in premises.items():
            print(" v ".join(value))
        print("\n")
        right_premises=premises.copy()
        instructions=read_instructions(sys.argv[3])
        for instruction in instructions:
            literal = instruction[0]
            what_to_do=instruction[1]
            if(what_to_do=="?"):
                lenght_premises=len(premises)
                print("User's command: {} {}".format(literal, what_to_do))
                path=[]

                right_premises = premises.copy()
              
                for key in premises:
                    premises[key] = set(premises[key])
                    useable_clauses_numbers = list(premises.keys())
                
                #prvo micemo nadskupe
                removed=set()
                for i in range(len(premises)):
                    for j in range(len(premises)):
                        if (i!=j):
                            if(premises[i+1].issubset(premises[j+1])):
                                removed.add(j+1)
                useable_clauses_numbers = [x for x in useable_clauses_numbers if x not in removed]
                #zatim brisemo tautologije
                for i in useable_clauses_numbers:
                    #isTaut = isTautology(premises[i+1])
                    if(isTautology(premises[i])):
                        useable_clauses_numbers.remove(i)
                old = set(useable_clauses_numbers)

                SoS=set()
                if(len(literal.split(" "))==1):

                    if(literal.startswith("~")):
                        premises[len(premises)+1]=[literal]
                    else:
                        premises[len(premises)+1]=["~"+literal]

                    SoS.add(len(premises))
                else:
                    parts = literal.split(" v ")
                    first = parts[0]
                    second = parts[1]
                    if(first.startswith("~")):
                        first = first.strip("~")
                    else:
                        first="~"+first
                    if(second.startswith("~")):
                        second = second.strip("~")
                    else:
                        second="~"+second
                    premises[len(premises)+1]=[first]
                    premises[len(premises)+1]=[second]
                    SoS.add(len(premises)-1)
                    SoS.add(len(premises))

                SoS_at_beggining = SoS.copy()
                    

                length = len(premises)

                isDeductive = findifNIL(premises, SoS, old, found)


                if(isDeductive):
                    path=path[::-1]
                    path_numbers=set()
                    for element in path:
                        path_numbers.add(element[1][0])
                        path_numbers.add(element[1][1])
                    for number in path_numbers:
                        if number<=lenght_premises or number in SoS_at_beggining:
                            print(str(number) + ". " + " v ".join(premises[number]))
                    print("=" * 15)
                    counter=length
                    for element in path:
                        counter+=1
                        if not element[0]:
                            print(str(counter) + ". NIL " + "(" + str(element[1][0]) + ", " + str(element[1][1]) + ")")
                        else:
                            print(str(counter) + ". " + " v ".join(element[0]) + " " + "(" + str(element[1][0]) + ", " + str(element[1][1]) + ")")
                    print("=" * 15)  
                    print("[CONCLUSION]: {} is true".format(literal))
                else:
                    print("=" * 15)
                    print("[CONCLUSION]: {} is unknown".format(literal))

                print("\n")

                premises=right_premises.copy()                

            elif(what_to_do=="+"):
                print("User's command: {} {}".format(literal, what_to_do))
                if(len(literal.split(" v "))>1):
                    premises[len(premises)+1]=literal.split(" v ")
                else:
                    premises[len(premises)+1]=[literal]
                print("Added {}".format(literal))
                print("\n")
            elif(what_to_do=="-"):
                print("User's command: {} {}".format(literal, what_to_do))
                deleted_key=""
                for key, value in premises.items():
                    if(len(literal.split(" v "))>1):
                        if value == literal.split(" v "):
                            del premises[key]
                            deleted_key = key
                            break
                    else:
                        if value == [literal]:
                            del premises[key]
                            deleted_key = key
                            break
                helper_var = {}
                for i in range(len(premises)):
                    if (i+1)<(deleted_key):

                        helper_var[i+1]=premises[i+1]
                    else:
                        helper_var[i+1]=premises[i+2]
                premises = helper_var.copy()
                print("Removed {}".format(literal))
                print("\n")

    else:
        print("kuharski asistent")
else:
    print("Pogreška")
    exit


