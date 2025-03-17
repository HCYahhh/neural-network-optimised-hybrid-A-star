"""
Hybrid A* with trailer
@author: Huiming Zhou
"""
"""
1.扩展节点的时候可以不采用C.MOVE_STEP，单次扩展中直接用较大的step和当前的车轮转角来计算下一节点位置，并用圆弧连接两点，在该圆弧上采样来确保路径分辨率
2.碰撞检测能否采用Voronoi 势场，加上偏航角与障碍物边缘走向的夹角惩罚
3.RS曲线的生成与检测碰撞太耗时
"""

import os
import sys
import math
import heapq
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial import KDTree

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../neural network optimised hybrid A-star/")

import HybridAstarPlanner.astar as astar
import HybridAstarPlanner.draw as draw
import CurvesGenerator.reeds_shepp as rs

t_collision=0          #所有碰撞检测耗时
t_rs_generate=0        #RS曲线生成，排序并离散耗时
t_hmap=0               #生成启发式代价地图时间


class C:  # Parameter config
    PI = np.pi

    XY_RESO = 2.0  # [m]
    YAW_RESO = np.deg2rad(15.0)  # [rad]
    GOAL_YAW_ERROR = np.deg2rad(3.0)  # [rad]
    MOVE_STEP = 0.2  # [m] path interporate resolution
    N_STEER = 5.0  # number of steer command
    COLLISION_CHECK_STEP = 5  # skip number for collision check
    EXTEND_AREA = 1.0  # [m] map extend length

    GEAR_COST = 100.0  # switch back penalty cost 前后方向改变惩罚
    BACKWARD_COST = 1.0  # backward penalty cost
    STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
    STEER_ANGLE_COST = 1.0  # steer angle penalty cost
    SCISSORS_COST = 50.0  # scissors cost 铰接角惩罚
    H_COST = 500.0  # Heuristic cost

    W = 2.438  # [m] width of vehicle
    WB = 4.135  # [m] wheel base: rear to front steer
    WD = 0.8 * W  # [m] distance between left-right wheels
    RF = 5.635  # [m] distance from rear to vehicle front end of vehicle 牵引车后轴到牵引车前端的距离
    RB = 1.365  # [m] distance from rear to vehicle back end of vehicle 牵引车后轴到牵引车后端的距离

    Ra=0.335#[m]铰接点（更靠近牵引车车头）到牵引车后轴的距离

    RTa = 7.9  # [m] 铰接点到挂车后轴距离
    FTa = 1.3  # [m] 铰接点到挂车前端距离
    BTa = 12.1  # [m] 铰接点到挂车后端距离
    TR = 0.5  # [m] tyre radius
    TW = 0.5  # [m] tyre width 
    MAX_STEER = 0.6  # [rad] maximum steering angle


class Node:
    def __init__(self, xind, yind, yawind, direction, x, y,
                 yaw, yawt, directions, steer, cost, pind):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.directions = directions
        self.steer = steer
        self.cost = cost
        self.pind = pind


class Para:
    def __init__(self, minx, miny, minyaw, minyawt, maxx, maxy, maxyaw, maxyawt,
                 xw, yw, yaww, yawtw, xyreso, yawreso, ox, oy, kdtree):
        self.minx = minx
        self.miny = miny
        self.minyaw = minyaw
        self.minyawt = minyawt
        self.maxx = maxx
        self.maxy = maxy
        self.maxyaw = maxyaw
        self.maxyawt = maxyawt
        self.xw = xw
        self.yw = yw
        self.yaww = yaww
        self.yawtw = yawtw
        self.xyreso = xyreso
        self.yawreso = yawreso
        self.ox = ox
        self.oy = oy
        self.kdtree = kdtree


class Path:
    def __init__(self, x, y, yaw, yawt, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.yawt = yawt
        self.direction = direction
        self.cost = cost


class QueuePrior:
    def __init__(self):
        self.queue = []

    def empty(self):
        return len(self.queue) == 0  # if Q is empty

    def put(self, item, priority):
        heapq.heappush(self.queue, (priority, item))  # reorder x using priority

    def get(self):
        return heapq.heappop(self.queue)[1]  # pop out element with smallest priority


def hybrid_astar_planning(sx, sy, syaw, syawt, gx, gy,
                          gyaw, gyawt, ox, oy, xyreso, yawreso):
    """
    planning hybrid A* path.
    :param sx: starting node x position [m]
    :param sy: starting node y position [m]
    :param syaw: starting node yaw angle [rad]
    :param syawt: starting node trailer yaw angle [rad]
    :param gx: goal node x position [m]
    :param gy: goal node y position [m]
    :param gyaw: goal node yaw angle [rad]
    :param gyawt: goal node trailer yaw angle [rad]
    :param ox: obstacle x positions [m]
    :param oy: obstacle y positions [m]
    :param xyreso: grid resolution [m]
    :param yawreso: yaw resolution [m]
    :return: hybrid A* path
    """

    sxr, syr = round(sx / xyreso), round(sy / xyreso)
    gxr, gyr = round(gx / xyreso), round(gy / xyreso)
    syawr = round(rs.pi_2_pi(syaw) / yawreso)
    gyawr = round(rs.pi_2_pi(gyaw) / yawreso)

    nstart = Node(sxr, syr, syawr, 1, [sx], [sy], [syaw], [syawt], [1], 0.0, 0.0, -1)
    ngoal = Node(gxr, gyr, gyawr, 1, [gx], [gy], [gyaw], [gyawt], [1], 0.0, 0.0, -1)

    kdtree = KDTree([[x, y] for x, y in zip(ox, oy)])
    P = calc_parameters(ox, oy, xyreso, yawreso, kdtree)

    global t_hmap
    t0=time.time()
    hmap = astar.calc_holonomic_heuristic_with_obstacle(ngoal, P.ox, P.oy, P.xyreso, 1.0)  #A星算启发式代价地图
    t1=time.time()
    t_hmap=t1-t0
    steer_set, direc_set = calc_motion_set()
    open_set, closed_set = {calc_index(nstart, P): nstart}, {}

    qp = QueuePrior()
    qp.put(calc_index(nstart, P), calc_hybrid_cost(nstart, hmap, P))

    while True:
        if not open_set:
            print("final expand node: ", len(open_set) + len(closed_set))
            return None,open_set,closed_set

        ind = qp.get()
        n_curr = open_set[ind]
        closed_set[ind] = n_curr
        open_set.pop(ind)

        update, fpath = update_node_with_analystic_expantion(n_curr, ngoal, gyawt, P)

        if update:
            fnode = fpath
            break

        yawt0 = n_curr.yawt[0]

        for i in range(len(steer_set)):
            node = calc_next_node(n_curr, ind, steer_set[i], direc_set[i], P)

            if not is_index_ok(node, yawt0, P,steer_set[i]):
                continue

            node_ind = calc_index(node, P)

            if node_ind in closed_set:
                continue

            if node_ind not in open_set:
                open_set[node_ind] = node
                qp.put(node_ind, calc_hybrid_cost(node, hmap, P))
            else:
                if open_set[node_ind].cost > node.cost:
                    open_set[node_ind] = node

    print("final expand node: ", len(open_set) + len(closed_set))

    return extract_path(closed_set, fnode, nstart),open_set,closed_set


def extract_path(closed, ngoal, nstart):
    rx, ry, ryaw, ryawt, direc = [], [], [], [], []
    cost = 0.0
    node = ngoal

    while True:
        rx += node.x[::-1]
        ry += node.y[::-1]
        ryaw += node.yaw[::-1]
        ryawt += node.yawt[::-1]
        direc += node.directions[::-1]
        cost += node.cost

        if is_same_grid(node, nstart):
            break

        node = closed[node.pind]

    rx = rx[::-1]
    ry = ry[::-1]
    ryaw = ryaw[::-1]
    ryawt = ryawt[::-1]
    direc = direc[::-1]

    direc[0] = direc[1]
    path = Path(rx, ry, ryaw, ryawt, direc, cost)

    return path


def update_node_with_analystic_expantion(n_curr, ngoal, gyawt, P):
    path,yawt,steps = analystic_expantion(n_curr, ngoal, P,gyawt)  # rs path: n -> ngoal

    if not path:
        return False, None

    fx = path.x[1:-1]
    fy = path.y[1:-1]
    fyaw = path.yaw[1:-1]
    fd = path.directions[1:-1]
    fcost = n_curr.cost + calc_rs_path_cost(path, yawt)
    fpind = calc_index(n_curr, P)
    fyawt = yawt[1:-1]
    fsteer = 0.0

    fpath = Node(n_curr.xind, n_curr.yind, n_curr.yawind, n_curr.direction,
                 fx, fy, fyaw, fyawt, fd, fsteer, fcost, fpind)

    return True, fpath


def analystic_expantion(node, ngoal, P,gyawt):
    global t_rs_generate
    t0 = time.time()
    sx, sy, syaw = node.x[-1], node.y[-1], node.yaw[-1]
    gx, gy, gyaw = ngoal.x[-1], ngoal.y[-1], ngoal.yaw[-1]

    maxc = math.tan(C.MAX_STEER) / C.WB  #计算最大曲率

    paths = rs.calc_all_paths(sx, sy, syaw, gx, gy, gyaw, maxc, step_size=C.MOVE_STEP)

    if not paths:
        t1 = time.time()
        t_rs_generate += (t1 - t0)
        return None,None,None

    pq = QueuePrior()
    for path in paths:
        steps = [C.MOVE_STEP * d for d in path.directions]
        yawt = calc_trailer_yaw(path.yaw, node.yawt[-1], steps,ctypes=path.ctypes)
        pq.put(path, calc_rs_path_cost(path, yawt))
    t1 = time.time()
    t_rs_generate += (t1 - t0)

    while not pq.empty():
        path = pq.get()
        steps = [C.MOVE_STEP * d for d in path.directions]
        yawt = calc_trailer_yaw(path.yaw, node.yawt[-1], steps,ctypes=path.ctypes)

        if abs(rs.pi_2_pi(yawt[-1] - gyawt)) <= C.GOAL_YAW_ERROR:

            ind = range(0, len(path.x), C.COLLISION_CHECK_STEP)

            pathx = [path.x[k] for k in ind]
            pathy = [path.y[k] for k in ind]
            pathyaw = [path.yaw[k] for k in ind]
            pathyawt = [yawt[k] for k in ind]

            if not is_collision(pathx, pathy, pathyaw, pathyawt, P):
                return path,yawt,steps

    return None,None,None


def calc_next_node(n, ind, u, d, P):
    step = C.XY_RESO * 2.0

    nlist = math.ceil(step / C.MOVE_STEP)
    xlist = [n.x[-1] + d * C.MOVE_STEP * math.cos(n.yaw[-1])]
    ylist = [n.y[-1] + d * C.MOVE_STEP * math.sin(n.yaw[-1])]
    yawlist = [rs.pi_2_pi(n.yaw[-1] + d * C.MOVE_STEP / C.WB * math.tan(u))]
    beta=math.atan(C.Ra*math.tan(u)/C.WB)
    yawtlist = [rs.pi_2_pi(n.yawt[-1] +
                           d * C.MOVE_STEP / C.RTa * math.sin(n.yaw[-1] - n.yawt[-1] + beta)/math.cos(beta))]

    for i in range(nlist - 1):
        xlist.append(xlist[i] + d * C.MOVE_STEP * math.cos(yawlist[i]))
        ylist.append(ylist[i] + d * C.MOVE_STEP * math.sin(yawlist[i]))
        yawlist.append(rs.pi_2_pi(yawlist[i] + d * C.MOVE_STEP / C.WB * math.tan(u)))
        yawtlist.append(rs.pi_2_pi(yawtlist[i] +
                                   d * C.MOVE_STEP / C.RTa * math.sin(yawlist[i] - yawtlist[i] + beta) / math.cos(beta)))

    xind = round(xlist[-1] / P.xyreso)
    yind = round(ylist[-1] / P.xyreso)
    yawind = round(yawlist[-1] / P.yawreso)

    cost = 0.0

    if d > 0:
        direction = 1.0
        cost += abs(step)
    else:
        direction = -1.0
        cost += abs(step) * C.BACKWARD_COST

    if direction != n.direction:  # switch back penalty
        cost += C.GEAR_COST

    cost += C.STEER_ANGLE_COST * abs(u)  # steer penalyty
    cost += C.STEER_CHANGE_COST * abs(n.steer - u)  # steer change penalty
    cost += C.SCISSORS_COST * sum([abs(rs.pi_2_pi(x - y))
                                   for x, y in zip(yawlist, yawtlist)])  # jacknif cost
    cost = n.cost + cost

    directions = [direction for _ in range(len(xlist))]

    node = Node(xind, yind, yawind, direction, xlist, ylist,
                yawlist, yawtlist, directions, u, cost, ind)

    return node


def is_collision(x, y, yaw, yawt, P):
    global t_collision
    t0=time.time()
    for ix, iy, iyaw, iyawt in zip(x, y, yaw, yawt):
        d = 0.5
        deltal = (C.FTa - C.BTa) / 2.0
        rt = (C.FTa + C.BTa) / 2.0 + d

        ctx = ix + C.Ra * math.cos(iyaw) + deltal * math.cos(iyawt)  #挂车碰撞圆中心
        cty = iy + C.Ra * math.sin(iyaw) + deltal * math.sin(iyawt)

        idst = P.kdtree.query_ball_point([ctx, cty], rt)

        if idst:
            for i in idst:
                xot = P.ox[i] - ctx
                yot = P.oy[i] - cty

                dx_trail = xot * math.cos(iyawt) + yot * math.sin(iyawt)
                dy_trail = -xot * math.sin(iyawt) + yot * math.cos(iyawt)

                if abs(dx_trail) <= rt and \
                        abs(dy_trail) <= C.W / 2.0 + d:
                    return True

        deltal = (C.RF - C.RB) / 2.0
        rc = (C.RF + C.RB) / 2.0 + d

        cx = ix + deltal * math.cos(iyaw)
        cy = iy + deltal * math.sin(iyaw)

        ids = P.kdtree.query_ball_point([cx, cy], rc)

        if ids:
            for i in ids:
                xo = P.ox[i] - cx
                yo = P.oy[i] - cy

                dx_car = xo * math.cos(iyaw) + yo * math.sin(iyaw)
                dy_car = -xo * math.sin(iyaw) + yo * math.cos(iyaw)

                if abs(dx_car) <= rc and \
                        abs(dy_car) <= C.W / 2.0 + d:
                    return True

    t1=time.time()
    t_collision+=(t1-t0)

    return False


def calc_trailer_yaw(yaw, yawt0, steps,steer_angle=0,ctypes=None):
    yawt = [0.0 for _ in range(len(yaw))]
    yawt[0] = yawt0

    for i in range(1, len(yaw)):
        if ctypes=="WB":
            beta=math.atan(C.Ra*math.tan(C.MAX_STEER)/C.WB)
        elif ctypes=="R":
            beta=math.atan(C.Ra*math.tan(-C.MAX_STEER)/C.WB)
        elif ctypes=="S":
            beta=0
        else:
            beta=math.atan(C.Ra*math.tan(steer_angle)/C.WB)

        yawt[i] += yawt[i - 1] + steps[i - 1] / C.RTa * math.sin(yaw[i - 1] - yawt[i - 1] + beta) / math.cos(beta)

    return yawt

def calc_rs_path_cost(rspath, yawt):
    cost = 0.0

    for lr in rspath.lengths:
        if lr >= 0:
            cost += 1
        else:
            cost += abs(lr) * C.BACKWARD_COST

    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
            cost += C.GEAR_COST

    for ctype in rspath.ctypes:
        if ctype != "S":
            cost += C.STEER_ANGLE_COST * abs(C.MAX_STEER)

    nctypes = len(rspath.ctypes)
    ulist = [0.0 for _ in range(nctypes)]

    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = -C.MAX_STEER
        elif rspath.ctypes[i] == "WB":
            ulist[i] = C.MAX_STEER

    for i in range(nctypes - 1):
        cost += C.STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    cost += C.SCISSORS_COST * sum([abs(rs.pi_2_pi(x - y))
                                   for x, y in zip(rspath.yaw, yawt)])

    return cost


def calc_motion_set():
    s = [i for i in np.arange(C.MAX_STEER / C.N_STEER,
                              C.MAX_STEER, C.MAX_STEER / C.N_STEER)]

    steer = [0.0] + s + [-i for i in s]
    direc = [1.0 for _ in range(len(steer))] + [-1.0 for _ in range(len(steer))]
    steer = steer + steer

    return steer, direc


def calc_hybrid_cost(node, hmap, P):
    cost = node.cost + \
           C.H_COST * hmap[node.xind - P.minx][node.yind - P.miny]

    return cost


def calc_index(node, P):
    ind = (node.yawind - P.minyaw) * P.xw * P.yw + \
          (node.yind - P.miny) * P.xw + \
          (node.xind - P.minx)

    yawt_ind = round(node.yawt[-1] / P.yawreso)
    ind += (yawt_ind - P.minyawt) * P.xw * P.yw * P.yaww

    return ind


def is_index_ok(node, yawt0, P,steer_angle):
    if node.xind <= P.minx or \
            node.xind >= P.maxx or \
            node.yind <= P.miny or \
            node.yind >= P.maxy:
        return False

    steps = [C.MOVE_STEP * d for d in node.directions]
    yawt = calc_trailer_yaw(node.yaw, yawt0, steps,steer_angle=steer_angle)

    # Check scissors angle constraint
    for yaw_i, yawt_i in zip(node.yaw, yawt):
        scissors_angle = abs(rs.pi_2_pi(yaw_i - yawt_i))
        if scissors_angle > math.pi / 2:  # 90 degree threshold
            return False

    ind = range(0, len(node.x), C.COLLISION_CHECK_STEP)

    x = [node.x[k] for k in ind]
    y = [node.y[k] for k in ind]
    yaw = [node.yaw[k] for k in ind]
    yawt = [yawt[k] for k in ind]

    if is_collision(x, y, yaw, yawt, P):
        return False

    return True


def calc_parameters(ox, oy, xyreso, yawreso, kdtree):
    minxm = min(ox) - C.EXTEND_AREA
    minym = min(oy) - C.EXTEND_AREA
    maxxm = max(ox) + C.EXTEND_AREA
    maxym = max(oy) + C.EXTEND_AREA

    ox.append(minxm)
    oy.append(minym)
    ox.append(maxxm)
    oy.append(maxym)

    minx = round(minxm / xyreso)
    miny = round(minym / xyreso)
    maxx = round(maxxm / xyreso)
    maxy = round(maxym / xyreso)

    xw, yw = maxx - minx, maxy - miny

    minyaw = round(-C.PI / yawreso) - 1
    maxyaw = round(C.PI / yawreso)
    yaww = maxyaw - minyaw

    minyawt, maxyawt, yawtw = minyaw, maxyaw, yaww

    P = Para(minx, miny, minyaw, minyawt, maxx, maxy, maxyaw,
             maxyawt, xw, yw, yaww, yawtw, xyreso, yawreso, ox, oy, kdtree)

    return P


def is_same_grid(node1, node2):
    if node1.xind != node2.xind or \
            node1.yind != node2.yind or \
            node1.yawind != node2.yawind:
        return False

    return True


def draw_model(x, y, yaw, yawt, steer, color='black'):
    car = np.array([[-C.RB, -C.RB, C.RF, C.RF, -C.RB],
                    [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    trail = np.array([[-C.BTa, -C.BTa, C.FTa, C.FTa, -C.BTa],
                      [C.W / 2, -C.W / 2, -C.W / 2, C.W / 2, C.W / 2]])

    wheel = np.array([[-C.TR, -C.TR, C.TR, C.TR, -C.TR],
                      [C.TW / 4, -C.TW / 4, -C.TW / 4, C.TW / 4, C.TW / 4]])

    rlWheel = wheel.copy()
    rrWheel = wheel.copy()
    frWheel = wheel.copy()
    flWheel = wheel.copy()
    rltWheel = wheel.copy()
    rrtWheel = wheel.copy()

    Rot1 = np.array([[math.cos(yaw), -math.sin(yaw)],
                     [math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), -math.sin(steer)],
                     [math.sin(steer), math.cos(steer)]])

    Rot3 = np.array([[math.cos(yawt), -math.sin(yawt)],
                     [math.sin(yawt), math.cos(yawt)]])

    frWheel = np.dot(Rot2, frWheel)
    flWheel = np.dot(Rot2, flWheel)

    frWheel += np.array([[C.WB], [-C.WD / 2]])
    flWheel += np.array([[C.WB], [C.WD / 2]])
    rrWheel[1, :] -= C.WD / 2
    rlWheel[1, :] += C.WD / 2

    frWheel = np.dot(Rot1, frWheel)
    flWheel = np.dot(Rot1, flWheel)

    rrWheel = np.dot(Rot1, rrWheel)
    rlWheel = np.dot(Rot1, rlWheel)
    car = np.dot(Rot1, car)

    rltWheel += np.array([[-C.RTa], [C.WD / 2]])
    rrtWheel += np.array([[-C.RTa], [-C.WD / 2]])

    rltWheel = np.dot(Rot3, rltWheel)
    rrtWheel = np.dot(Rot3, rrtWheel)
    trail = np.dot(Rot3, trail)

    frWheel += np.array([[x], [y]])
    flWheel += np.array([[x], [y]])
    rrWheel += np.array([[x], [y]])
    rlWheel += np.array([[x], [y]])
    rrtWheel += np.array([[x], [y]])
    rltWheel += np.array([[x], [y]])
    car += np.array([[x], [y]])
    trail += np.array([[x], [y]])

    plt.plot(car[0, :], car[1, :], color)
    plt.plot(trail[0, :], trail[1, :], color)
    plt.plot(frWheel[0, :], frWheel[1, :], color)
    plt.plot(rrWheel[0, :], rrWheel[1, :], color)
    plt.plot(flWheel[0, :], flWheel[1, :], color)
    plt.plot(rlWheel[0, :], rlWheel[1, :], color)
    plt.plot(rrtWheel[0, :], rrtWheel[1, :], color)
    plt.plot(rltWheel[0, :], rltWheel[1, :], color)
    draw.Arrow(x, y, yaw, C.WB * 0.8, color)


def design_obstacles():
    ox, oy = [], []
    for i in range(0, 25 + random.randint(-1, 1)):
        for j in range(20 + random.randint(-1, 1), 30 + random.randint(-1, 1)):
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
        oy.append(40)
    for j in range(0, 40):
        ox.append(0)
        oy.append(j)
        ox.append(40)
        oy.append(j)
    return ox, oy


def update(frame, path, ox, oy,open_coords,closed_coords):
    plt.cla()  # 清空当前帧
    plt.plot(ox, oy, 'sk')  # 绘制障碍物
    plt.plot(path.x[:frame+1], path.y[:frame+1], 'r', linewidth=1.5) # 绘制路径（到当前帧）

    # # 绘制开放列表（蓝色）和关闭列表（红色）
    # if len(open_coords[0]) > 0:
    #     plt.scatter(open_coords[0], open_coords[1], c='blue', s=2, label='Open List')
    # if len(closed_coords[0]) > 0:
    #     plt.scatter(closed_coords[0], closed_coords[1], c='red', s=2, label='Closed List')

    # 绘制车辆模型
    if frame < len(path.x):
        x, y = path.x[frame], path.y[frame]
        yaw = path.yaw[frame]
        yawt = path.yawt[frame]
        direction = path.direction[frame]

        # 计算转向角（基于前后帧的yaw变化）
        if frame < len(path.x) - 1:
            dy = path.yaw[frame + 1] - yaw
            steer = np.arctan(C.WB * dy / (C.MOVE_STEP * direction))
        else:
            steer = 0.0

        draw_model(x, y, yaw, yawt, steer)

    plt.title(f"Frame: {frame}/{len(path.x)}")
    plt.axis("equal")
    return []

def extract_node_coordinates(node_set):
    x, y = [], []
    for node in node_set.values():
        x += node.x
        y += node.y
    return x, y

def main():
    global t_collision
    global t_rs_generate
    global t_hmap
    print("start!")

    # 初始化参数
    sx, sy = 18.0,36.0
    syaw0 = np.deg2rad(0.0)
    syawt = np.deg2rad(0.0)
    gx, gy = 18.0, 3.0
    gyaw0 = np.deg2rad(0.0)
    gyawt = np.deg2rad(0.0)
    ox, oy = design_obstacles()

    # 路径规划
    t0 = time.time()
    path,open_set,closed_set = hybrid_astar_planning(sx, sy, syaw0, syawt, gx, gy, gyaw0, gyawt,
                                 ox, oy, C.XY_RESO, C.YAW_RESO)
    t1 = time.time()
    print("sum running T: ", t1 - t0)
    print('t_collision:',t_collision)
    print('t_rs_generate:', t_rs_generate)
    print('t_hmap',t_hmap)
    if path is None:
        print("Path not found!")
        return

    # 提取开放列表和关闭列表的坐标
    open_x, open_y = extract_node_coordinates(open_set)
    closed_x, closed_y = extract_node_coordinates(closed_set)

    # 设置画布
    plt.figure(figsize=(10, 8))
    plt.plot(ox, oy, 'sk', label="Obstacles")
    draw_model(sx, sy, syaw0, syawt, 0.0)
    draw_model(gx, gy, gyaw0, gyawt, 0.0, 'gray')

    # 创建动画
    ani = FuncAnimation(
        plt.gcf(),                                  # 获取当前Figure对象
        lambda frame: update(frame, path, ox, oy,
        (open_x, open_y), (closed_x, closed_y)),  # 更新函数
        frames=len(path.x),                         # 总帧数（路径点数）
        interval=50,                                # 帧间隔时间（ms），控制动画速度
        blit=False,                                 # 是否使用blit优化（建议关闭，避免复杂图形问题）
        repeat=False                               # 动画结束后不重复播放
    )

    plt.legend()
    plt.show()
    print("Done")


if __name__ == '__main__':
    main()
