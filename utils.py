from random import randint


def ManualFibreLossModel(key1, key2, numNodes, fibreLen=0, iniLoss=0.2, lenLoss=0.25, algorithmFator=2):
    """
    Model fibre loss after keys were formed to avoid disrupting the BB84 flow.
    """
    keyLen = len(key1)
    lossCount = 0

    # length-based loss
    if fibreLen != 0:
        for _ in range(int(fibreLen)):
            for _ in range(keyLen):
                myrand = randint(0, 100)
                if myrand < lenLoss * 100:
                    lossCount += 1

    # initial loss at each hop
    for _ in range(numNodes - 1):
        myrand = randint(0, 100)
        if myrand < iniLoss * 100:
            lossCount += 1

    lossCount /= algorithmFator

    if lossCount >= keyLen:
        return [], []

    new_len = keyLen - int(lossCount)
    return key1[:new_len], key2[:new_len]


def Random_basis_gen(length):
    """
    Return a random list of 0/1 basis choices of the given length.
    """
    return [randint(0, 1) for _ in range(length)]


def CompareBasis(loc_basis_list, rem_basis_list, loc_res):
    """
    Remove entries where bases do not match; return the filtered results.
    """
    if len(loc_basis_list) != len(rem_basis_list):
        print("Comparing error! length of basis does not match!")
        return -1

    popList = []
    for i in range(len(rem_basis_list)):
        if loc_basis_list[i] != rem_basis_list[i]:
            popList.append(i)

    for i in reversed(popList):
        if loc_res:
            loc_res.pop(i)

    return loc_res
