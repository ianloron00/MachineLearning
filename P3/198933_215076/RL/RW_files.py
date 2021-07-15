# variable that allows or not reading and writing values from/to files.
CAN_READ_WRITE_FILES=False

def checkIfRead():
    print ("Enter y to read weights from file, n otherwise")
    option = input()
    if option == "y": return True
    return False

def read_file(state):
    walls=state.getWalls()
    size=len(walls[0])
    title="weights "
    if size==7: title+="smallClassic.txt"
    elif size==11: title+="mediumClassic.txt"
    else: title+="originalClassic.txt"
    file = open(title, "r")
    dict = {}
    for line in file.readlines():
        f, val  = line.strip().split()
        dict[f] = float(val)
    file.close()
    return dict
        
def checkIfSave():
    print ("Enter y to save file, n otherwise")
    option = input()
    if option == "y": return True
    return False

def save_file(state,dict):
    walls=state.getWalls()
    size=len(walls[0])
    title="weights "
    if size==7: title+="smallClassic.txt"
    elif size==11: title+="mediumClassic.txt"
    else: title+="originalClassic.txt"
    file = open(title, "w")
    for f in dict:
        file.write(f+" "+str(dict[f])+"\n")
    file.close()