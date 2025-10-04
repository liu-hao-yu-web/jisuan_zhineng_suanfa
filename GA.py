import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False

# ==================== 测试函数定义 ====================
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


# ==================== 遗传算法实现 ====================
class 遗传算法:
    def __init__(self, 目标函数, 种群大小=30, 最大迭代次数=100, 维度=2, 范围=(-5, 5),
                 交叉率=0.8, 变异率=0.1):
        self.目标函数 = 目标函数
        self.种群大小 = 种群大小
        self.最大迭代次数 = 最大迭代次数
        self.维度 = 维度
        self.范围 = 范围
        self.交叉率 = 交叉率
        self.变异率 = 变异率

        # 记录优化过程
        self.历史位置 = []
        self.历史最优解 = []
        self.历史最优值 = []

    def 初始化种群(self):
        """初始化种群"""
        self.种群位置 = np.random.uniform(
            self.范围[0], self.范围[1], (self.种群大小, self.维度))

        # 计算初始适应度
        self.种群适应度 = np.array([self.目标函数(个体) for 个体 in self.种群位置])

        # 记录全局最优
        self.全局最优索引 = np.argmin(self.种群适应度)
        self.全局最优解 = self.种群位置[self.全局最优索引].copy()
        self.全局最优值 = self.种群适应度[self.全局最优索引]

    def 选择操作(self):
        """轮盘赌选择"""
        # 将适应度转换为选择概率（最小化问题需要取倒数）
        适应度倒数 = 1 / (self.种群适应度 + 1e-10)  # 加上小值避免除零
        选择概率 = 适应度倒数 / np.sum(适应度倒数)

        # 根据概率选择个体
        选择索引 = np.random.choice(
            range(self.种群大小), size=self.种群大小, p=选择概率, replace=True)

        self.种群位置 = self.种群位置[选择索引]
        self.种群适应度 = self.种群适应度[选择索引]

    def 交叉操作(self):
        """模拟二进制交叉"""
        新种群 = []

        for i in range(0, self.种群大小, 2):
            if i + 1 >= self.种群大小:
                新种群.append(self.种群位置[i])
                break

            父代1 = self.种群位置[i]
            父代2 = self.种群位置[i + 1]

            if np.random.rand() < self.交叉率:
                # 模拟二进制交叉
                子代1 = np.zeros(self.维度)
                子代2 = np.zeros(self.维度)

                for j in range(self.维度):
                    if np.random.rand() < 0.5:
                        # 模拟二进制交叉
                        β = np.random.rand()
                        子代1[j] = 0.5 * ((1 + β) * 父代1[j] + (1 - β) * 父代2[j])
                        子代2[j] = 0.5 * ((1 - β) * 父代1[j] + (1 + β) * 父代2[j])
                    else:
                        子代1[j] = 父代1[j]
                        子代2[j] = 父代2[j]

                # 边界检查
                子代1 = np.clip(子代1, self.范围[0], self.范围[1])
                子代2 = np.clip(子代2, self.范围[0], self.范围[1])

                新种群.append(子代1)
                新种群.append(子代2)
            else:
                新种群.append(父代1)
                新种群.append(父代2)

        self.种群位置 = np.array(新种群)

    def 变异操作(self):
        """高斯变异"""
        for i in range(self.种群大小):
            for j in range(self.维度):
                if np.random.rand() < self.变异率:
                    # 高斯变异
                    变异量 = np.random.randn() * 0.1 * (self.范围[1] - self.范围[0])
                    self.种群位置[i, j] += 变异量

            # 边界检查
            self.种群位置[i] = np.clip(self.种群位置[i], self.范围[0], self.范围[1])

    def 更新适应度(self):
        """更新种群适应度"""
        self.种群适应度 = np.array([self.目标函数(个体) for 个体 in self.种群位置])

        # 更新全局最优
        当前最优索引 = np.argmin(self.种群适应度)
        当前最优值 = self.种群适应度[当前最优索引]

        if 当前最优值 < self.全局最优值:
            self.全局最优索引 = 当前最优索引
            self.全局最优解 = self.种群位置[当前最优索引].copy()
            self.全局最优值 = 当前最优值

    def 运行优化(self):
        """执行优化过程"""
        self.初始化种群()

        # 记录初始状态
        self.历史位置.append(self.种群位置.copy())
        self.历史最优解.append(self.全局最优解.copy())
        self.历史最优值.append(self.全局最优值)

        for 迭代次数 in range(self.最大迭代次数):
            self.选择操作()
            self.交叉操作()
            self.变异操作()
            self.更新适应度()

            # 记录当前状态
            self.历史位置.append(self.种群位置.copy())
            self.历史最优解.append(self.全局最优解.copy())
            self.历史最优值.append(self.全局最优值)

            if 迭代次数 % 10 == 0:
                print(f"迭代次数: {迭代次数}, 当前最优值: {self.全局最优值:.6f}")

        print(f"优化完成! 最优解: {self.全局最优解}, 最优值: {self.全局最优值:.6f}")
        return self.全局最优解, self.全局最优值


# ==================== 动画可视化 ====================
class 遗传算法可视化器:
    def __init__(self, 算法实例):
        self.算法实例 = 算法实例

        # 准备函数曲面数据
        self.x = np.linspace(算法实例.范围[0], 算法实例.范围[1], 100)
        self.y = np.linspace(算法实例.范围[0], 算法实例.范围[1], 100)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = np.array([ackley函数([x, y]) for x, y in zip(np.ravel(self.X), np.ravel(self.Y))]).reshape(
            self.X.shape)

    def 创建动画(self, 文件名=None):
        """创建算法优化过程动画"""
        fig, ax = plt.subplots(figsize=(10, 8))

        # 设置图形属性
        ax.set_xlim(self.算法实例.范围[0], self.算法实例.范围[1])
        ax.set_ylim(self.算法实例.范围[0], self.算法实例.范围[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('遗传算法 - Ackley函数优化过程')

        # 绘制函数曲面
        contour = ax.contourf(self.X, self.Y, self.Z, 50, cmap=cm.viridis, alpha=0.8)
        fig.colorbar(contour, ax=ax, label='函数值')

        # 初始化种群散点
        散点图 = ax.scatter([], [], c='red', s=30, alpha=0.7, label='种群个体')
        最优解点 = ax.scatter([], [], c='blue', s=100, marker='*', label='全局最优解')
        ax.legend()

        # 添加文本显示迭代信息
        迭代文本 = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # 动画更新函数
        def 更新动画帧(帧):
            if 帧 >= len(self.算法实例.历史位置):
                return 散点图, 最优解点, 迭代文本

            # 更新种群位置
            当前位置 = self.算法实例.历史位置[帧]
            散点图.set_offsets(当前位置)

            # 更新全局最优解位置
            当前最优解 = self.算法实例.历史最优解[帧]
            最优解点.set_offsets([当前最优解])

            # 更新迭代信息
            迭代文本.set_text(f'迭代次数: {帧}/{self.算法实例.最大迭代次数}\n'
                              f'当前最优值: {self.算法实例.历史最优值[帧]:.6f}')

            return 散点图, 最优解点, 迭代文本

        # 创建动画
        动画 = FuncAnimation(fig, 更新动画帧, frames=len(self.算法实例.历史位置),
                             interval=200, blit=True, repeat=True)

        plt.tight_layout()

        # 保存动画
        if 文件名:
            动画.save(文件名, writer='pillow', fps=5)

        plt.show()

        return 动画


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 创建遗传算法实例
    ga = 遗传算法(ackley函数, 种群大小=30, 最大迭代次数=50, 维度=2, 范围=(-5, 5))

    # 运行优化
    开始时间 = time.time()
    最优解, 最优值 = ga.运行优化()
    结束时间 = time.time()
    print(f"优化耗时: {结束时间 - 开始时间:.2f}秒")

    # 创建动画
    可视化器 = 遗传算法可视化器(ga)
    动画 = 可视化器.创建动画("遗传算法_优化过程.gif")

    # 绘制收敛曲线
    plt.figure(figsize=(10, 6))
    plt.plot(ga.历史最优值)
    plt.xlabel('迭代次数')
    plt.ylabel('最优值')
    plt.title('遗传算法收敛曲线')
    plt.grid(True)
    plt.savefig('遗传算法收敛曲线.png', dpi=300, bbox_inches='tight')
    plt.show()