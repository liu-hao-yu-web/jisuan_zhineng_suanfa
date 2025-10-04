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


# ==================== 进化编程实现 ====================
class 进化编程算法:
    def __init__(self, 目标函数, 种群大小=30, 最大迭代次数=100, 维度=2, 范围=(-5, 5),
                 初始变异强度=0.5, 学习率=0.1, 竞争参数=10):
        self.目标函数 = 目标函数
        self.种群大小 = 种群大小
        self.最大迭代次数 = 最大迭代次数
        self.维度 = 维度
        self.范围 = 范围
        self.初始变异强度 = 初始变异强度
        self.学习率 = 学习率
        self.竞争参数 = 竞争参数  # 竞争选择中的竞争对手数量

        # 记录优化过程
        self.历史位置 = []
        self.历史变异强度 = []
        self.历史最优解 = []
        self.历史最优值 = []

    def 初始化种群(self):
        """初始化种群和变异强度"""
        # 初始化种群位置
        self.种群位置 = np.random.uniform(
            self.范围[0], self.范围[1], (self.种群大小, self.维度))

        # 初始化变异强度（每个个体每个维度都有自己的变异强度）
        self.变异强度 = np.ones((self.种群大小, self.维度)) * self.初始变异强度

        # 计算初始适应度
        self.种群适应度 = np.array([self.目标函数(个体) for 个体 in self.种群位置])

        # 记录全局最优
        self.全局最优索引 = np.argmin(self.种群适应度)
        self.全局最优解 = self.种群位置[self.全局最优索引].copy()
        self.全局最优值 = self.种群适应度[self.全局最优索引]

    def 变异操作(self):
        """基于变异强度的高斯变异 - 每个个体产生一个子代"""
        子代位置 = np.zeros((self.种群大小, self.维度))
        子代变异强度 = np.zeros((self.种群大小, self.维度))

        for i in range(self.种群大小):
            # 变异强度的自适应性更新
            子代变异强度[i] = self.变异强度[i] * np.exp(
                self.学习率 * np.random.randn(self.维度))

            # 确保变异强度不会太小
            子代变异强度[i] = np.maximum(子代变异强度[i], 0.01)

            # 基于变异强度的高斯变异
            子代位置[i] = self.种群位置[i] + 子代变异强度[i] * np.random.randn(self.维度)

            # 边界检查
            子代位置[i] = np.clip(子代位置[i], self.范围[0], self.范围[1])

        return 子代位置, 子代变异强度

    def 竞争选择(self, 父代位置, 父代变异强度, 父代适应度, 子代位置, 子代变异强度):
        """竞争选择 - 从父代和子代中选择最好的个体"""
        # 合并父代和子代
        合并位置 = np.vstack((父代位置, 子代位置))
        合并变异强度 = np.vstack((父代变异强度, 子代变异强度))

        # 计算所有个体的适应度
        合并适应度 = np.hstack((父代适应度, np.array([self.目标函数(个体) for 个体 in 子代位置])))

        # 初始化新种群
        新种群位置 = np.zeros((self.种群大小, self.维度))
        新种群变异强度 = np.zeros((self.种群大小, self.维度))
        新种群适应度 = np.zeros(self.种群大小)

        # 对每个个体进行竞争选择
        for i in range(len(合并位置)):
            # 随机选择q个竞争对手
            竞争对手索引 = np.random.choice(range(len(合并位置)), self.竞争参数, replace=False)

            # 计算赢得的比赛次数
            获胜次数 = np.sum(合并适应度[i] < 合并适应度[竞争对手索引])

            # 如果赢得的比赛次数足够多，选择该个体
            if 获胜次数 >= self.竞争参数 / 2:
                if np.sum(新种群适应度 == 0) > 0:
                    空位索引 = np.where(新种群适应度 == 0)[0][0]
                    新种群位置[空位索引] = 合并位置[i]
                    新种群变异强度[空位索引] = 合并变异强度[i]
                    新种群适应度[空位索引] = 合并适应度[i]

        # 如果新种群未填满，用表现最好的个体填充
        空位数量 = np.sum(新种群适应度 == 0)
        if 空位数量 > 0:
            # 按适应度排序
            排序索引 = np.argsort(合并适应度)
            最佳个体索引 = 排序索引[:空位数量]

            空位索引 = np.where(新种群适应度 == 0)[0]
            新种群位置[空位索引] = 合并位置[最佳个体索引]
            新种群变异强度[空位索引] = 合并变异强度[最佳个体索引]
            新种群适应度[空位索引] = 合并适应度[最佳个体索引]

        return 新种群位置, 新种群变异强度, 新种群适应度

    def 运行优化(self):
        """执行优化过程"""
        self.初始化种群()

        # 记录初始状态
        self.历史位置.append(self.种群位置.copy())
        self.历史变异强度.append(self.变异强度.copy())
        self.历史最优解.append(self.全局最优解.copy())
        self.历史最优值.append(self.全局最优值)

        for 迭代次数 in range(self.最大迭代次数):
            # 变异操作生成子代
            子代位置, 子代变异强度 = self.变异操作()

            # 竞争选择更新种群
            self.种群位置, self.变异强度, self.种群适应度 = self.竞争选择(
                self.种群位置, self.变异强度, self.种群适应度,
                子代位置, 子代变异强度)

            # 更新全局最优
            当前最优索引 = np.argmin(self.种群适应度)
            当前最优值 = self.种群适应度[当前最优索引]

            if 当前最优值 < self.全局最优值:
                self.全局最优索引 = 当前最优索引
                self.全局最优解 = self.种群位置[当前最优索引].copy()
                self.全局最优值 = 当前最优值

            # 记录当前状态
            self.历史位置.append(self.种群位置.copy())
            self.历史变异强度.append(self.变异强度.copy())
            self.历史最优解.append(self.全局最优解.copy())
            self.历史最优值.append(self.全局最优值)

            if 迭代次数 % 10 == 0:
                print(f"迭代次数: {迭代次数}, 当前最优值: {self.全局最优值:.6f}, "
                      f"平均变异强度: {np.mean(self.变异强度):.4f}")

        print(f"优化完成! 最优解: {self.全局最优解}, 最优值: {self.全局最优值:.6f}")
        return self.全局最优解, self.全局最优值


# ==================== 动画可视化 ====================
class 进化编程可视化器:
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 设置第一个子图（函数曲面和种群分布）
        ax1.set_xlim(self.算法实例.范围[0], self.算法实例.范围[1])
        ax1.set_ylim(self.算法实例.范围[0], self.算法实例.范围[1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('进化编程算法 - Ackley函数优化过程')

        # 绘制函数曲面
        contour = ax1.contourf(self.X, self.Y, self.Z, 50, cmap=cm.viridis, alpha=0.8)
        fig.colorbar(contour, ax=ax1, label='函数值')

        # 初始化粒子散点（大小表示变异强度）
        散点图 = ax1.scatter([], [], c='red', s=30, alpha=0.7, label='种群个体')
        最优解点 = ax1.scatter([], [], c='blue', s=100, marker='*', label='全局最优解')
        ax1.legend()

        # 设置第二个子图（收敛曲线和变异强度曲线）
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('最优值', color='blue')
        ax2.set_title('收敛曲线和变异强度变化')
        ax2.tick_params(axis='y', labelcolor='blue')

        # 创建第二个y轴用于显示变异强度
        ax2_右侧 = ax2.twinx()
        ax2_右侧.set_ylabel('平均变异强度', color='red')
        ax2_右侧.tick_params(axis='y', labelcolor='red')

        收敛曲线, = ax2.plot([], [], 'b-', label='最优值')
        变异强度曲线, = ax2_右侧.plot([], [], 'r-', label='平均变异强度')

        # 添加图例
        曲线列表 = [收敛曲线, 变异强度曲线]
        标签列表 = ['最优值', '平均变异强度']
        ax2.legend(曲线列表, 标签列表, loc='upper right')

        # 添加文本显示迭代信息
        迭代文本 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, fontsize=12,
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        # 动画更新函数
        def 更新动画帧(帧):
            if 帧 >= len(self.算法实例.历史位置):
                return 散点图, 最优解点, 收敛曲线, 变异强度曲线, 迭代文本

            # 更新种群位置
            当前位置 = self.算法实例.历史位置[帧]
            散点图.set_offsets(当前位置)

            # 更新散点大小（反映变异强度）
            当前变异强度 = self.算法实例.历史变异强度[帧]
            散点大小 = 30 * np.mean(当前变异强度, axis=1) / self.算法实例.初始变异强度
            散点图.set_sizes(散点大小)

            # 更新全局最优解位置
            当前最优解 = self.算法实例.历史最优解[帧]
            最优解点.set_offsets([当前最优解])

            # 更新收敛曲线
            收敛曲线.set_data(range(帧 + 1), self.算法实例.历史最优值[:帧 + 1])
            ax2.set_xlim(0, len(self.算法实例.历史位置))
            ax2.set_ylim(0, max(self.算法实例.历史最优值) if self.算法实例.历史最优值 else 1)

            # 更新变异强度曲线
            平均变异强度历史 = [np.mean(强度) for 强度 in self.算法实例.历史变异强度[:帧 + 1]]
            变异强度曲线.set_data(range(帧 + 1), 平均变异强度历史)
            ax2_右侧.set_ylim(0, max(平均变异强度历史) if 平均变异强度历史 else 1)

            # 更新迭代信息
            迭代文本.set_text(f'算法: 进化编程\n'
                              f'迭代次数: {帧}/{self.算法实例.最大迭代次数}\n'
                              f'当前最优值: {self.算法实例.历史最优值[帧]:.6f}\n'
                              f'平均变异强度: {np.mean(当前变异强度):.4f}\n'
                              f'竞争参数: {self.算法实例.竞争参数}')

            return 散点图, 最优解点, 收敛曲线, 变异强度曲线, 迭代文本

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
    # 创建进化编程算法实例
    ep = 进化编程算法(ackley函数, 种群大小=30, 最大迭代次数=50, 维度=2,
                      范围=(-5, 5), 初始变异强度=0.5, 学习率=0.1, 竞争参数=10)

    # 运行优化
    开始时间 = time.time()
    最优解, 最优值 = ep.运行优化()
    结束时间 = time.time()
    print(f"优化耗时: {结束时间 - 开始时间:.2f}秒")

    # 创建动画
    可视化器 = 进化编程可视化器(ep)
    动画 = 可视化器.创建动画("进化编程_优化过程.gif")

    # 绘制收敛曲线和变异强度变化
    plt.figure(figsize=(12, 5))

    # 收敛曲线
    plt.subplot(1, 2, 1)
    plt.plot(ep.历史最优值)
    plt.xlabel('迭代次数')
    plt.ylabel('最优值')
    plt.title('进化编程收敛曲线')
    plt.grid(True)

    # 变异强度变化
    plt.subplot(1, 2, 2)
    平均变异强度历史 = [np.mean(强度) for 强度 in ep.历史变异强度]
    plt.plot(平均变异强度历史, 'r-')
    plt.xlabel('迭代次数')
    plt.ylabel('平均变异强度')
    plt.title('变异强度自适应变化')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('进化编程收敛曲线和变异强度.png', dpi=300, bbox_inches='tight')
    plt.show()