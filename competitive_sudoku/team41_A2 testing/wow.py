wow = [{2, 1, 3}, {1, 3, 7}, {1, 4, 8}, {4, 8}, {5, 0}]
def present_only_once(wow: list, x: int) -> bool:
    check = 0
    for element in wow:
        if (x in element):
            check += 1
            if (check == 2):
                return(False)
    return(True)
print(present_only_once(wow, 3))