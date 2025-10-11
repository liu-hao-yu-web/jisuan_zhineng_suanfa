import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import pandas as pd
import opfunu
import json
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100  



CEC2005理论最优值 = {
    'F12005': -450,
    'F22005': -450,
    'F32005': -450,
    'F42005': -450,
    'F52005': -310,
    'F62005': 390,
    'F72005': -180,
    'F82005': -140,
    'F92005': -330,
    'F102005': -330,
    'F112005': 90,
    'F122005': -460,
    'F132005': -130,
    'F142005': -300,
    'F152005': 120,
    'F162005': 120,
    'F172005': 120,
    'F182005': 10,
    'F192005': 10,
    'F202005': 10,
    'F212005': 360,
    'F222005': 360,
    'F232005': 360,
    'F242005': 260,
    'F252005': 260
}

#=================================定义粒子群优化算法类================================
class 粒子群优化算法:
    def __init__(self, 目标函数, 种群大小=30, 最大迭代次数=100, 维度=2, 范围=(-5, 5),
                 惯性权重=0.7, 个体学习因子=1.5, 社会学习因子=1.5,收敛阈值=1e-8):
        self.目标函数 = 目标函数
        self.种群大小 = 种群大小
        self.最大迭代次数 = 最大迭代次数
        self.维度 = 维度
        self.范围 = 范围
        self.惯性权重 = 惯性权重
        self.个体学习因子 = 个体学习因子
        self.社会学习因子 = 社会学习因子

        
        self.粒子位置 = np.random.uniform(范围[0], 范围[1], (种群大小, 维度))
        self.粒子速度 = np.zeros((种群大小, 维度))

        
        self.个体最优位置 = self.粒子位置.copy()
        self.个体最优值 = np.array([目标函数(个体) for 个体 in self.粒子位置])
        self.全局最优索引 = np.argmin(self.个体最优值)
        self.全局最优位置 = self.个体最优位置[self.全局最优索引].copy()
        self.全局最优值 = self.个体最优值[self.全局最优索引]

        
        self.历史位置 = [self.粒子位置.copy()]
        self.历史全局最优 = [self.全局最优位置.copy()]
        self.历史最优值 = [self.全局最优值]
        self.收敛阈值 = 收敛阈值

    def 更新种群(self):
        
        for i in range(self.种群大小):
            
            随机因子1 = np.random.rand(self.维度)
            随机因子2 = np.random.rand(self.维度)

            
            认知分量 = self.个体学习因子 * 随机因子1 * (self.个体最优位置[i] - self.粒子位置[i])
            社会分量 = self.社会学习因子 * 随机因子2 * (self.全局最优位置 - self.粒子位置[i])
            self.粒子速度[i] = self.惯性权重 * self.粒子速度[i] + 认知分量 + 社会分量

            
            self.粒子位置[i] += self.粒子速度[i]

            
            self.粒子位置[i] = np.clip(self.粒子位置[i], self.范围[0], self.范围[1])

            
            当前值 = self.目标函数(self.粒子位置[i])

            
            if 当前值 < self.个体最优值[i]:
                self.个体最优位置[i] = self.粒子位置[i].copy()
                self.个体最优值[i] = 当前值

                
                if 当前值 < self.全局最优值:
                    self.全局最优位置 = self.粒子位置[i].copy()
                    self.全局最优值 = 当前值

        
        self.历史位置.append(self.粒子位置.copy())
        self.历史全局最优.append(self.全局最优位置.copy())
        self.历史最优值.append(self.全局最优值)
        
        if self.收敛阈值 is not None and hasattr(self, '理论最优值'):
            当前误差 = abs(self.全局最优值 - self.理论最优值)
            if 当前误差 < self.收敛阈值:
                
                self.已收敛 = True

    def 运行优化(self, 理论最优值=None):
        
        
        self.理论最优值 = 理论最优值

        
        self.开始时间 = time()

        
        self.收敛代数 = None
        self.收敛时间 = None

        for 迭代次数 in range(self.最大迭代次数):
            self.更新种群()

            
            if self.收敛阈值 is not None and self.理论最优值 is not None:
                当前误差 = abs(self.全局最优值 - self.理论最优值)
                if 当前误差 < self.收敛阈值:
                    self.收敛代数 = 迭代次数
                    self.收敛时间 = time() - self.开始时间
                    print(f"在迭代 {迭代次数} 收敛，误差: {当前误差:.2e}")
                    break  

        
        
        if self.收敛代数 is None:
            self.收敛代数 = self.最大迭代次数
            self.收敛时间 = time() - self.开始时间

        
        总FES = self.种群大小 * (self.收敛代数 + 1)  

        print(f"优化完成! 最优值: {self.全局最优值:.6e}, "
              f"收敛代数: {self.收敛代数}, 总FES: {总FES}")

        return (self.全局最优位置, self.全局最优值,
                self.收敛代数, self.收敛时间, 总FES)
#==================== 批量测试函数 ====================

def 批量测试CEC2005函数(算法参数=None):
    if 算法参数 is None:
        
        算法参数 = {
            '种群大小': 50,
            '最大迭代次数': 5000,
            '惯性权重': 0.7,  
            '个体学习因子': 1.5,  
            '社会学习因子': 1.5,  
            '收敛阈值': 1e-8
        }
    
    CEC2005函数列表 = [f'F{i}2005' for i in range(1, 26)]
    
    支持维度列表 = [10]
    运行次数 = 25  

    
    固定精度水平 = {
        'F12005': 1e-8, 'F22005': 1e-8, 'F32005': 1e-8, 'F42005': 1e-8, 'F52005': 1e-8,
        'F62005': 1e-2, 'F72005': 1e-2, 'F82005': 1e-2, 'F92005': 1e-2, 'F102005': 1e-2,
        'F112005': 1e-2, 'F122005': 1e-2, 'F132005': 1e-2, 'F142005': 1e-2, 'F152005': 1e-2,
        'F162005': 1e-2, 'F172005': 1e-2, 'F182005': 1e-2, 'F192005': 1e-2, 'F202005': 1e-2,
        'F212005': 1e-2, 'F222005': 1e-2, 'F232005': 1e-2, 'F242005': 1e-2, 'F252005': 1e-2
    }

    
    if not os.path.exists('PSO的CEC2005测试结果'):
        os.makedirs('PSO的CEC2005测试结果')

    if not os.path.exists('PSO的CEC2005测试结果/收敛曲线'):
        os.makedirs('PSO的CEC2005测试结果/收敛曲线')

    所有结果 = {}
    详细结果 = {}

    print("开始批量测试PSO的CEC2005函数（遵循CEC 2005评估标准）...")
    print(f"测试函数数量: {len(CEC2005函数列表)}")
    print(f"测试维度: {支持维度列表}")
    print(f"每个组合运行次数: {运行次数}")
    print("=" * 80)
    for 函数名 in CEC2005函数列表:
        print(f"\n正在测试函数: {函数名}")

        函数结果 = {}
        函数详细结果 = {}

        for 维度 in 支持维度列表:
            print(f"  维度: {维度}", end=" | ")

            
            try:
                测试函数列表 = opfunu.get_functions_by_classname(函数名)
                if not 测试函数列表:
                    print(f"警告: 未找到函数 {函数名}，跳过")
                    continue

                具体测试函数 = 测试函数列表[0](ndim=维度)
                下界 = 具体测试函数.lb[0]
                上界 = 具体测试函数.ub[0]
                理论最优值 = CEC2005理论最优值.get(函数名, 0)
                目标精度 = 固定精度水平.get(函数名, 1e-8)

                def 目标函数(x):
                    return 具体测试函数.evaluate(x)

            except Exception as e:
                print(f"初始化函数出错: {e}")
                continue

            
            最终误差列表 = []
            收敛时间列表 = []
            收敛代数列表 = []
            函数评估次数列表 = []
            成功运行次数 = 0
            达到精度的FES列表 = []

            
            FES1000误差列表 = []
            FES10000误差列表 = []
            FES100000误差列表 = []

            for 运行编号 in range(运行次数):
                
                
                粒子群优化算法实例 = 粒子群优化算法(
                    目标函数=目标函数,
                    种群大小=算法参数['种群大小'],
                    最大迭代次数=算法参数['最大迭代次数'],
                    维度=维度,
                    范围=(下界, 上界),
                    惯性权重=算法参数.get('惯性权重', 0.7),  
                    个体学习因子=算法参数.get('个体学习因子', 1.5),  
                    社会学习因子=算法参数.get('社会学习因子', 1.5),  

                )

                
                最优解, 最优值, 收敛代数, 收敛时间, 总FES = 粒子群优化算法实例.运行优化(理论最优值)

                
                真实误差 = abs(最优值 - 理论最优值)
                最终误差列表.append(真实误差)
                收敛时间列表.append(收敛时间)
                收敛代数列表.append(收敛代数)
                函数评估次数列表.append(总FES)

                
                if 真实误差 <= 目标精度:
                    成功运行次数 += 1
                    达到精度的FES列表.append(总FES)

                
                if hasattr(粒子群优化算法实例, '函数评估次数记录'):
                    
                    FES记录 = 粒子群优化算法实例.函数评估次数记录
                    最优值记录 = 粒子群优化算法实例.历史最优值

                    
                    idx_1000 = np.searchsorted(FES记录, 1000, side='right') - 1
                    if idx_1000 >= 0:
                        FES1000误差列表.append(abs(最优值记录[idx_1000] - 理论最优值))

                    
                    idx_10000 = np.searchsorted(FES记录, 10000, side='right') - 1
                    if idx_10000 >= 0:
                        FES10000误差列表.append(abs(最优值记录[idx_10000] - 理论最优值))

                    
                    idx_100000 = np.searchsorted(FES记录, 100000, side='right') - 1
                    if idx_100000 >= 0:
                        FES100000误差列表.append(abs(最优值记录[idx_100000] - 理论最优值))

                if (运行编号 + 1) % 5 == 0:
                    print(f"{运行编号 + 1}", end=" ")

            
            排序后误差 = np.sort(最终误差列表)
            排序后FES = np.sort(达到精度的FES列表) if 达到精度的FES列表 else np.array([])

            统计结果 = {
                '理论最优值': 理论最优值,
                '目标精度': 目标精度,
                '平均误差': float(np.mean(最终误差列表)),
                '误差标准差': float(np.std(最终误差列表)),
                '最佳误差': float(np.min(最终误差列表)),
                '最差误差': float(np.max(最终误差列表)),
                '平均收敛时间': float(np.mean(收敛时间列表)),
                '平均收敛代数': float(np.mean(收敛代数列表)),
                '平均FES': float(np.mean(函数评估次数列表)),
                '成功运行次数': 成功运行次数,
                '成功率': 成功运行次数 / 运行次数,
                '排序后误差': 排序后误差.tolist(),
                '排序后FES': 排序后FES.tolist(),
                'FES1000误差': float(np.mean(FES1000误差列表)) if FES1000误差列表 else None,
                'FES10000误差': float(np.mean(FES10000误差列表)) if FES10000误差列表 else None,
                'FES100000误差': float(np.mean(FES100000误差列表)) if FES100000误差列表 else None
            }

            函数结果[维度] = 统计结果
            print(f"| 平均误差: {统计结果['平均误差']:.2e}, 成功率: {统计结果['成功率']:.2%}")

        所有结果[函数名] = 函数结果

        
        详细结果[函数名] = {
            '所有运行误差': 最终误差列表,
            '所有运行时间': 收敛时间列表,
            '所有运行FES': 函数评估次数列表,
            '达到精度的FES': 达到精度的FES列表
        }

    return 所有结果, 详细结果
#==================== 结果分析与可视化 ====================
def 生成CEC标准报告(所有结果, 详细结果):
    
    print("\n" + "=" * 100)
    print("CEC 2005标准测试报告")
    print("=" * 100)

    
    汇总数据 = []

    for 函数名, 维度结果 in 所有结果.items():
        for 维度, 统计结果 in 维度结果.items():
            排序后误差 = 统计结果['排序后误差']
            误差统计 = {
                '函数名': 函数名,
                '维度': 维度,
                '理论最优值': 统计结果['理论最优值'],
                '最佳误差(1st)': 排序后误差[0] if len(排序后误差) > 0 else None,
                '第7佳误差': 排序后误差[6] if len(排序后误差) > 6 else None,
                '中位误差(13th)': 排序后误差[12] if len(排序后误差) > 12 else None,
                '第19佳误差': 排序后误差[18] if len(排序后误差) > 18 else None,
                '最差误差(25th)': 排序后误差[24] if len(排序后误差) > 24 else None,
                '平均误差': 统计结果['平均误差'],
                '误差标准差': 统计结果['误差标准差'],
                '成功率': 统计结果['成功率'],
                'FES1000误差': 统计结果['FES1000误差'],
                'FES10000误差': 统计结果['FES10000误差'],
                'FES100000误差': 统计结果['FES100000误差']
            }
            汇总数据.append(误差统计)

    
    汇总报告 = pd.DataFrame(汇总数据)

    
    汇总报告.to_excel('PSO的CEC2005测试结果/PSO的CEC2005_详细报告.xlsx', index=False)
    汇总报告.to_csv('PSO的CEC2005测试结果/PSO的CEC2005_详细报告.csv', index=False, encoding='utf-8-sig')

    
    with open('PSO的CEC2005测试结果/PSO的CEC2005_原始数据.json', 'w', encoding='utf-8') as f:
        json.dump(所有结果, f, ensure_ascii=False, indent=2)

    
    print("\n各维度性能摘要:")
    维度摘要 = 汇总报告.groupby('维度').agg({
        '平均误差': ['mean', 'std'],
        '成功率': 'mean',
        '函数名': 'count'
    }).round(4)
    print(维度摘要)

    print("\n各函数类型性能摘要:")
    
    函数类型映射 = {
        **{f'F{i}2005': '单峰函数' for i in range(1, 6)},
        **{f'F{i}2005': '多峰函数' for i in range(6, 25)}
    }
    汇总报告['函数类型'] = 汇总报告['函数名'].map(函数类型映射)

    if '函数类型' in 汇总报告.columns:
        类型摘要 = 汇总报告.groupby('函数类型').agg({
            '平均误差': ['mean', 'std'],
            '成功率': 'mean',
            '函数名': 'count'
        }).round(4)
        print(类型摘要)

    return 汇总报告


def 绘制CEC收敛曲线(详细结果, 选择函数=None, 选择维度=10):
    
    if 选择函数 is None:
        选择函数 = ['F12005', 'F62005', 'F92005', 'F152005', 'F212005']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, 函数名 in enumerate(选择函数[:6]):
        if 函数名 in 详细结果:
            
            误差数据 = 详细结果[函数名]['所有运行误差']

            
            axes[i].boxplot(误差数据, showmeans=True)
            axes[i].set_title(f'{函数名} - 误差分布')
            axes[i].set_ylabel('误差（对数坐标）')
            axes[i].set_yscale('log')
            axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('PSO的CEC2005测试结果/收敛曲线/误差分布对比.png', dpi=300, bbox_inches='tight')
    plt.show()


def 绘制性能对比雷达图(所有结果, 维度=10):
    
    函数列表 = [f'F{i}2005' for i in range(1, 26)]
    性能指标 = []

    for 函数名 in 函数列表:
        if 函数名 in 所有结果 and 维度 in 所有结果[函数名]:
            性能指标.append(所有结果[函数名][维度]['平均误差'])
        else:
            性能指标.append(np.nan)

    
    性能指标 = np.array(性能指标)
    最大误差 = np.nanmax(性能指标)
    最小误差 = np.nanmin(性能指标)
    归一化指标 = (性能指标 - 最小误差) / (最大误差 - 最小误差 + 1e-10)

    
    角度 = np.linspace(0, 2 * np.pi, len(函数列表), endpoint=False).tolist()
    角度 += 角度[:1]  

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    
    归一化指标列表 = 归一化指标.tolist()
    归一化指标列表 += 归一化指标列表[:1]  
    ax.plot(角度, 归一化指标列表, 'o-', linewidth=2)
    ax.fill(角度, 归一化指标列表, alpha=0.25)

    
    ax.set_xticks(角度[:-1])
    ax.set_xticklabels(函数列表)

    
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1)

    
    plt.title(f'PSO的CEC2005函数在{维度}维下的性能对比雷达图', size=16, color='blue', y=1.1)

    
    ax.legend(['归一化平均误差'], loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig(f'PSO的CEC2005测试结果/维度{维度}性能雷达图.png', dpi=300, bbox_inches='tight')
    plt.show()

    return 归一化指标


def 绘制维度影响分析(所有结果, 选择函数='F12005'):
    
    if 选择函数 not in 所有结果:
        print(f"函数 {选择函数} 不在结果中")
        return

    维度列表 = []
    误差列表 = []
    时间列表 = []
    成功率列表 = []

    for 维度, 统计结果 in 所有结果[选择函数].items():
        维度列表.append(维度)
        误差列表.append(统计结果['平均误差'])
        时间列表.append(统计结果['平均收敛时间'])
        成功率列表.append(统计结果['成功率'])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    
    ax1.plot(维度列表, 误差列表, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('维度')
    ax1.set_ylabel('平均误差（对数坐标）')
    ax1.set_yscale('log')
    ax1.set_title(f'{选择函数} - 误差随维度变化')
    ax1.grid(True, alpha=0.3)

    
    ax2.plot(维度列表, 时间列表, 's-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('维度')
    ax2.set_ylabel('平均收敛时间(s)')
    ax2.set_title(f'{选择函数} - 时间随维度变化')
    ax2.grid(True, alpha=0.3)

    
    ax3.plot(维度列表, 成功率列表, '^-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('维度')
    ax3.set_ylabel('成功率')
    ax3.set_title(f'{选择函数} - 成功率随维度变化')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'PSO的CEC2005测试结果/{选择函数}维度影响分析.png', dpi=300, bbox_inches='tight')
    plt.show()


def 绘制函数类型对比(所有结果, 维度=10):
    
    
    函数类型映射 = {
        **{f'F{i}2005': '单峰函数' for i in range(1, 6)},
        **{f'F{i}2005': '多峰函数' for i in range(6, 25)}
    }

    单峰函数误差 = []
    多峰函数误差 = []

    for 函数名, 维度结果 in 所有结果.items():
        if 维度 in 维度结果:
            函数类型 = 函数类型映射.get(函数名, '其他')
            if 函数类型 == '单峰函数':
                单峰函数误差.append(维度结果[维度]['平均误差'])
            elif 函数类型 == '多峰函数':
                多峰函数误差.append(维度结果[维度]['平均误差'])

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    
    误差数据 = [单峰函数误差, 多峰函数误差]
    ax1.boxplot(误差数据, tick_labels=['单峰函数', '多峰函数'], showmeans=True)
    ax1.set_ylabel('平均误差（对数坐标）')
    ax1.set_yscale('log')
    ax1.set_title('函数类型误差对比')
    ax1.grid(True, alpha=0.3)

    
    平均误差 = [np.mean(单峰函数误差), np.mean(多峰函数误差)]
    误差标准差 = [np.std(单峰函数误差), np.std(多峰函数误差)]

    bars = ax2.bar(['单峰函数', '多峰函数'], 平均误差, yerr=误差标准差,
                   capsize=5, alpha=0.7, color=['blue', 'orange'])
    ax2.set_ylabel('平均误差（对数坐标）')
    ax2.set_yscale('log')
    ax2.set_title('函数类型平均误差对比')
    ax2.grid(True, alpha=0.3)

    
    for bar, 误差 in zip(bars, 平均误差):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height * 1.05,
                 f'{误差:.2e}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'PSO的CEC2005测试结果/维度{维度}函数类型对比.png', dpi=300, bbox_inches='tight')
    plt.show()


def 生成性能排名表(所有结果, 维度=10):
    
    排名数据 = []

    for 函数名, 维度结果 in 所有结果.items():
        if 维度 in 维度结果:
            排名数据.append({
                '函数名': 函数名,
                '平均误差': 维度结果[维度]['平均误差'],
                '成功率': 维度结果[维度]['成功率'],
                '收敛时间': 维度结果[维度]['平均收敛时间']
            })

    
    排名表 = pd.DataFrame(排名数据)
    排名表 = 排名表.sort_values('平均误差')
    排名表['排名'] = range(1, len(排名表) + 1)

    
    排名表.to_excel(f'PSO的CEC2005测试结果/维度{维度}性能排名.xlsx', index=False)
    排名表.to_csv(f'PSO的CEC2005测试结果/维度{维度}性能排名.csv', index=False, encoding='utf-8-sig')

    print(f"\n维度 {维度} 性能排名前10:")
    print(排名表.head(10).to_string(index=False))

    return 排名表



#==================== 主程序 ====================
if __name__ == "__main__":
    aaa=time()
    
    算法参数配置 = {
        '种群大小': 2000,
        '最大迭代次数': 125,
        '惯性权重': 0.7,  
        '个体学习因子': 1.5,  
        '社会学习因子': 1.5,  
        '收敛阈值': 1e-8
    }

    
    print("开始CEC2005函数批量测试...")
    开始时间 = time()
    所有结果, 详细结果 = 批量测试CEC2005函数(算法参数配置)
    总耗时 = time() - 开始时间

    print(f"\n批量测试完成! 总耗时: {总耗时:.2f}秒")

    
    汇总报告 = 生成CEC标准报告(所有结果, 详细结果)

    
    for 维度 in [10]:
        绘制性能对比雷达图(所有结果, 维度)
        生成性能排名表(所有结果, 维度)

    
    绘制维度影响分析(所有结果, 'F12005')
    绘制维度影响分析(所有结果, 'F62005')
    绘制维度影响分析(所有结果, 'F92005')

    
    绘制函数类型对比(所有结果, 10)

    
    绘制CEC收敛曲线(详细结果)

    print(f"\n所有结果已保存到 'PSO的CEC2005测试结果' 目录中")
    print(f"测试报告: PSO的CEC2005测试结果/PSO的CEC2005_详细报告.xlsx")
    print(f"原始数据: PSO的CEC2005测试结果/PSO的CEC2005_原始数据.json")

    bbb=time()
    print("花费时间",(bbb-aaa))
