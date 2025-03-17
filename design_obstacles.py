import random
MAP_LIST = []

def map_0():#非50x50
    ox, oy = [], []
    for i in range(-22, -4):
        for j in range(0, 23+random.randint(0,1)):
            ox.append(i)
            oy.append(j)
    for i in range(5, 30):
        for j in range(0, 26):
            ox.append(i)
            oy.append(j)
    for i in range(-22, 30):
        for j in range(38, 40):
            ox.append(i)
            oy.append(j)
    for i in range(-22, 30):
        for j in range(-2, 0):
            ox.append(i)
            oy.append(j)
    for j in range(-2, 40):
        ox.append(-23)
        oy.append(j)
        ox.append(30)
        oy.append(j)
    return ox, oy
MAP_LIST.append(map_0())

def map_1():#单次运行2-5分钟
    ox, oy = [], []
    for i in range(0, 10 + random.randint(-1, 1)):
        for j in range(0, 5 + random.randint(-2, 2)):
            ox.append(i)
            oy.append(j)
    for i in range(40 + random.randint(-1, 1), 50):
        for j in range(0, 5 + random.randint(-2, 2)):
            ox.append(i)
            oy.append(j)
    for i in range(25 + random.randint(-1, 1), 50):
        for j in range(16 + random.randint(-1, 1), 20 + random.randint(-1, 1)):
            ox.append(i)
            oy.append(j)
    for i in range(0, 25 + random.randint(-1, 1)):
        for j in range(34 + random.randint(-1, 1), 39 + random.randint(-1, 1)):
            ox.append(i)
            oy.append(j)

    for i in range(0, 50):
        ox.append(i)
        oy.append(0)
        ox.append(i)
        oy.append(50)
    for j in range(0, 50):
        ox.append(0)
        oy.append(j)
        ox.append(50)
        oy.append(j)
    return ox, oy
MAP_LIST.append(map_0())

def map_2():
    ox, oy = [], []
    for i in range(0, 30 + random.randint(-1, 1)):
        for j in range(20 + random.randint(-1, 1), 33 + random.randint(-1, 1)):
            ox.append(i)
            oy.append(j)
    for i in range(25 + random.randint(-1, 1), 40):
        for j in range(0, 10 + random.randint(-1, 1)):
            ox.append(i)
            oy.append(j)

    for i in range(0, 40):
        ox.append(i)
        oy.append(0)
        ox.append(i)
        oy.append(50)
    for j in range(0, 40):
        ox.append(0)
        oy.append(j)
        ox.append(50)
        oy.append(j)
    return ox, oy
MAP_LIST.append(map_2())