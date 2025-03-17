import os
import sys
import gym
from gym import spaces
import numpy as np
from HybridAstarPlanner.hybrid_astar_with_trailer import hybrid_astar_planning, C, design_obstacles
from design_obstacles import MAP_LIST

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../neural network optimised hybrid A-star/")


class HybridAstarEnv(gym.Env):
    def __init__(self):
        super(HybridAstarEnv, self).__init__()
        # 加载所有预定义地图
        self.maps = MAP_LIST
        self.map_size = (50, 50)  # 假设地图尺寸为50x50
        # 定义状态空间
        self.observation_space = spaces.Dict({
            "map": spaces.Box(low=0, high=1, shape=self.map_size, dtype=np.float32),
            "start_goal": spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        })
        # 定义动作空间
        self.action_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def reset(self):
        # 随机选择一个地图
        self.map_idx = np.random.randint(0, len(self.maps))
        self.ox, self.oy = self.maps[self.map_idx]
        # 随机生成起点和终点（需避开障碍物）
        self.sx, self.sy = self._random_free_position()
        self.gx, self.gy = self._random_free_position(exclude=(self.sx, self.sy))
        return self._get_observation()

    def step(self, action):
        # 更新权重参数
        self._update_weights(action)
        # 运行混合A*算法
        path, _, _ = hybrid_astar_planning(
            self.sx, self.sy, 0, 0, self.gx, self.gy, 0, 0,
            self.ox, self.oy, C.XY_RESO, C.YAW_RESO
        )
        # 计算奖励
        reward = self._calculate_reward(path)
        done = (path is not None)
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # 栅格化地图
        map_grid = np.zeros(self.map_size, dtype=np.float32)
        for x, y in zip(self.ox, self.oy):
            ix, iy = int(x), int(y)
            if 0 <= ix < self.map_size[0] and 0 <= iy < self.map_size[1]:
                map_grid[ix, iy] = 1.0
        # 归一化起点/终点坐标到[-1,1]
        start_goal = np.array([
            self.sx / self.map_size[0] * 2 - 1,
            self.sy / self.map_size[1] * 2 - 1,
            self.gx / self.map_size[0] * 2 - 1,
            self.gy / self.map_size[1] * 2 - 1
        ], dtype=np.float32)
        return {"map": map_grid, "start_goal": start_goal}

    def _update_weights(self, action):
        # 将动作映射到权重范围
        C.GEAR_COST = action[0] * 200
        C.BACKWARD_COST = action[1] * 10
        C.STEER_CHANGE_COST = action[2] * 10
        C.STEER_ANGLE_COST = action[3] * 5
        C.SCISSORS_COST = action[4] * 300
        C.H_COST = action[5] * 3000

    def _calculate_reward(self, path):
        if path is None:
            return -100  # 规划失败惩罚
        # 路径长度
        length = sum(np.hypot(np.diff(path.x), np.diff(path.y)))
        # 曲率方差（二阶导数近似）
        curvature = np.abs(np.diff(path.yaw, 2)).mean()
        # 搜索节点数（假设path.x的长度为节点数）
        nodes = len(path.x)
        # 综合奖励
        return -(0.1 * length + 1.0 * curvature + 0.05 * nodes)

    def _random_free_position(self, exclude=None):
        # 在地图自由区域随机生成坐标
        while True:
            x = np.random.uniform(0, self.map_size[0])
            y = np.random.uniform(0, self.map_size[1])
            if self._is_position_free(x, y) and (x, y) != exclude:
                return x, y

    def _is_position_free(self, x, y):
        # 检查坐标是否在障碍物外
        for ox, oy in zip(self.ox, self.oy):
            if int(ox) == int(x) and int(oy) == int(y):
                return False
        return True