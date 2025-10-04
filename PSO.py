import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import time
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False
# 定义测试函数 - 二维Ackley函数
def ackley函数(个体位置):
    """
    计算二维Ackley函数值
    参数:
        个体位置: 二维坐标点 [x, y]
    返回:
        函数值
    """
    x, y = 个体位置
    项1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
    项2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    结果 = 项1 + 项2 + 20 + np.e
    return 结果


# 定义粒子群优化算法类
class 粒子群优化算法:
    def __init__(self, 目标函数, 种群大小=30, 最大迭代次数=100, 维度=2, 范围=(-5, 5),
                 惯性权重=0.7, 个体学习因子=1.5, 社会学习因子=1.5):
        self.目标函数 = 目标函数
        self.种群大小 = 种群大小
        self.最大迭代次数 = 最大迭代次数
        self.维度 = 维度
        self.范围 = 范围
        self.惯性权重 = 惯性权重
        self.个体学习因子 = 个体学习因子
        self.社会学习因子 = 社会学习因子

        # 初始化种群
        self.粒子位置 = np.random.uniform(范围[0], 范围[1], (种群大小, 维度))
        self.粒子速度 = np.zeros((种群大小, 维度))

        # 记录个体最优和全局最优
        self.个体最优位置 = self.粒子位置.copy()
        self.个体最优值 = np.array([目标函数(个体) for 个体 in self.粒子位置])
        self.全局最优索引 = np.argmin(self.个体最优值)
        self.全局最优位置 = self.个体最优位置[self.全局最优索引].copy()
        self.全局最优值 = self.个体最优值[self.全局最优索引]

        # 记录历史数据用于动画
        self.历史位置 = [self.粒子位置.copy()]
        self.历史全局最优 = [self.全局最优位置.copy()]
        self.历史最优值 = [self.全局最优值]

    def 更新种群(self):
        """更新粒子位置和速度"""
        for i in range(self.种群大小):
            # 生成随机因子
            随机因子1 = np.random.rand(self.维度)
            随机因子2 = np.random.rand(self.维度)

            # 更新速度
            认知分量 = self.个体学习因子 * 随机因子1 * (self.个体最优位置[i] - self.粒子位置[i])
            社会分量 = self.社会学习因子 * 随机因子2 * (self.全局最优位置 - self.粒子位置[i])
            self.粒子速度[i] = self.惯性权重 * self.粒子速度[i] + 认知分量 + 社会分量

            # 更新位置
            self.粒子位置[i] += self.粒子速度[i]

            # 边界检查
            self.粒子位置[i] = np.clip(self.粒子位置[i], self.范围[0], self.范围[1])

            # 评估新位置
            当前值 = self.目标函数(self.粒子位置[i])

            # 更新个体最优
            if 当前值 < self.个体最优值[i]:
                self.个体最优位置[i] = self.粒子位置[i].copy()
                self.个体最优值[i] = 当前值

                # 更新全局最优
                if 当前值 < self.全局最优值:
                    self.全局最优位置 = self.粒子位置[i].copy()
                    self.全局最优值 = 当前值

        # 记录当前状态
        self.历史位置.append(self.粒子位置.copy())
        self.历史全局最优.append(self.全局最优位置.copy())
        self.历史最优值.append(self.全局最优值)

    def 运行优化(self):
        """执行优化过程"""
        for 迭代次数 in range(self.最大迭代次数):
            self.更新种群()
            if 迭代次数 % 10 == 0:
                print(f"迭代次数: {迭代次数}, 当前最优值: {self.全局最优值:.6f}")

        print(f"优化完成! 最优解: {self.全局最优位置}, 最优值: {self.全局最优值:.6f}")
        return self.全局最优位置, self.全局最优值


# 创建优化器实例
优化器 = 粒子群优化算法(ackley函数, 种群大小=30, 最大迭代次数=50)

# 运行优化
开始时间 = time.time()
最优解, 最优值 = 优化器.运行优化()
结束时间 = time.time()
print(f"优化耗时: {结束时间 - 开始时间:.2f}秒")

# 准备动画数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = ackley函数([X[i, j], Y[i, j]])

# 创建动画
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('粒子群优化算法 - Ackley函数优化过程')

# 绘制函数曲面
contour = ax.contourf(X, Y, Z, 50, cmap=cm.viridis, alpha=0.8)
fig.colorbar(contour, ax=ax, label='函数值')

# 初始化粒子散点
散点图 = ax.scatter([], [], c='red', s=30, alpha=0.7, label='粒子')
最优解点 = ax.scatter([], [], c='blue', s=100, marker='*', label='全局最优解')

# 添加文本显示迭代信息
迭代文本 = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

ax.legend()


# 动画更新函数
def 更新动画帧(帧):
    """更新动画的每一帧"""
    if 帧 >= len(优化器.历史位置):
        return 散点图, 最优解点, 迭代文本

    # 更新粒子位置
    当前位置 = 优化器.历史位置[帧]
    散点图.set_offsets(当前位置)

    # 更新全局最优解位置
    当前最优解 = 优化器.历史全局最优[帧]
    最优解点.set_offsets([当前最优解])

    # 更新迭代信息
    迭代文本.set_text(f'迭代次数: {帧}/{优化器.最大迭代次数}\n'
                      f'当前最优值: {优化器.历史最优值[帧]:.6f}')

    return 散点图, 最优解点, 迭代文本


# 创建动画
动画 = FuncAnimation(fig, 更新动画帧, frames=len(优化器.历史位置),
                     interval=200, blit=True, repeat=True)

plt.tight_layout()
plt.show()

# 如果需要保存动画
动画.save('粒子群优化算法_ackley函数.gif', writer='pillow', fps=5)