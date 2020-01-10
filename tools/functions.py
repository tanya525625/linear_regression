

def sign(el: float):
    if el > 0:
        return 1
    elif el == 0:
        return 0
    else:
        return -1


def ones(size):
    new_list = []
    for i in range(size):
        new_list.append(1)

    return new_list


def rmse(y_pred: list, y_true: list):
    rmse = 0
    n =  len(y_pred)

    for i in range(n):
        rmse += (y_true[i] - y_pred[i]) ** 2

    return (rmse / n) ** (1 / 2)



