import math


def find_area_nikita(figure):
    x_mid = sum([it[0] for it in figure]) // len(figure)
    y_mid = sum([it[1] for it in figure]) // len(figure)
    area = 0
    for k in range(-1, len(figure) - 1):
        point1 = figure[k]
        point2 = figure[k + 1]
        a = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        b = math.sqrt((point2[0] - x_mid) ** 2 + (point2[1] - y_mid) ** 2)
        c = math.sqrt((point1[0] - x_mid) ** 2 + (point1[1] - y_mid) ** 2)
        p = (a + b + c) / 2
        area += math.sqrt(p * (p - a) * (p - b) * (p - c))
    return round(area, 5)


def find_area_roman(figure):
    area = 0
    for i in range(len(figure)):
        area += figure[i - 1][0] * figure[i][1] - figure[i - 1][1] * figure[i][0]
    return 0.5 * abs(area)


def neighbourhood(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
