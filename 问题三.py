import numpy as np
import cmath
import math
from scipy.optimize import fsolve

# 常数定义
n1 = 1.0  # 空气折射率
theta1_deg = 10.0  # 入射角(度)
theta1 = np.deg2rad(theta1_deg)  # 转换为弧度
max_reflections = 5  # 最大反射光次数

# 衬底折射率计算函数
def n_substrate(lamb, polarization):
    """
    计算衬底对s或p偏振光的折射率
    
    参数:
    lamb: 波长(μm)
    polarization: 's' 或 'p'
    
    返回:
    衬底折射率
    """
    return 3.4169

    if polarization == 's':
        A = 5.67
        B = 0.034
    elif polarization == 'p':
        A = 5.59
        B = 0.029
    else:
        raise ValueError("偏振类型必须是's'或'p'")
    
    return math.sqrt(1 + (A * lamb**2) / (lamb**2 - B))

# 振幅反射率和透射率计算
def calculate_amplitudes(n1, n2, n3_s, n3_p, theta1, lamb):
    """
    计算所有振幅反射率和透射率
    
    参数:
    n1: 空气折射率
    n2: 外延层折射率
    n3_s: 衬底对s偏振的折射率
    n3_p: 衬底对p偏振的折射率
    theta1: 入射角(弧度)
    lamb: 波长
    
    返回:
    包含所有振幅反射率和透射率的字典
    """
    sin_theta1 = math.sin(theta1)
    cos_theta1 = math.cos(theta1)
    
    # 计算中间变量
    n1_sin_theta1_sq = (n1 * sin_theta1)**2
    k = math.sqrt(n2**2 - n1_sin_theta1_sq)  # 用于简化表达式
    
    # s偏振振幅
    rs12 = (n1 * cos_theta1 - k) / (n1 * cos_theta1 + k)
    ts12 = (2 * n1 * cos_theta1) / (n1 * cos_theta1 + k)
    
    k_s3 = math.sqrt(n3_s**2 - n1_sin_theta1_sq)
    rs23 = (k - k_s3) / (k + k_s3)
    rs21 = (k - n1 * cos_theta1) / (k + n1 * cos_theta1)
    ts21 = (2 * k) / (k + n1 * cos_theta1)
    
    # p偏振振幅
    rp12 = (n2**2 * cos_theta1 - n1 * k) / (n2**2 * cos_theta1 + n1 * k)
    tp12 = (2 * n1 * n2 * cos_theta1) / (n2**2 * cos_theta1 + n1 * k)
    
    k_p3 = math.sqrt(n3_p**2 - n1_sin_theta1_sq)
    rp23 = (k_p3 - k) / (k_p3 + k)
    rp21 = (n1 * cos_theta1 - k) / (n1 * cos_theta1 + k)
    
    # 计算θ2
    sin_theta2 = n1 * sin_theta1 / n2
    cos_theta2 = math.sqrt(1 - sin_theta2**2)
    tp21 = (2 * n2 * cos_theta2) / (n1 * cos_theta2 + n2 * cos_theta1)
    
    return {
        'rs12': rs12, 'ts12': ts12,
        'rp12': rp12, 'tp12': tp12,
        'rs23': rs23, 'rp23': rp23,
        'rs21': rs21, 'rp21': rp21,
        'ts21': ts21, 'tp21': tp21,
        'cos_theta2': cos_theta2
    }

# 计算多光束干涉的反射率
def calculate_reflectivity(n2, d, lamb, max_reflections=5):
    """
    计算多光束干涉的总反射率
    
    参数:
    n2: 外延层折射率
    d: 外延层厚度
    lamb: 波长
    max_reflections: 考虑的最大反射光次数
    
    返回:
    总反射率R
    """
    # 计算衬底折射率
    n3_s = n_substrate(lamb, 's')
    n3_p = n_substrate(lamb, 'p')
    
    # 计算所有振幅反射率和透射率
    amps = calculate_amplitudes(n1, n2, n3_s, n3_p, theta1, lamb)
    cos_theta2 = amps['cos_theta2']
    
    # 计算相位差参数
    beta = 4 * np.pi * n2 * d * cos_theta2 / lamb
    
    # 入射光振幅 (无偏振光，s和p分量相等)
    As = Ap = 1.0 / math.sqrt(2)  # 归一化
    
    # 初始化复振幅
    Er_s = 0 + 0j  # s偏振总反射振幅
    Er_p = 0 + 0j  # p偏振总反射振幅
    
    # 第一束反射光 (直接反射)
    # 相位为0
    Er_s += As * amps['rs12']
    Er_p += Ap * amps['rp12']
    
    # 后续反射光
    for k in range(2, max_reflections + 1):
        # 计算振幅
        amp_factor_s = amps['ts12'] * amps['ts21'] * amps['rs23'] * (amps['rs21'] * amps['rs23'])**(k-2)
        amp_factor_p = amps['tp12'] * amps['tp21'] * amps['rp23'] * (amps['rp21'] * amps['rp23'])**(k-2)
        
        A_ks = As * amp_factor_s
        A_kp = Ap * amp_factor_p
        
        # 计算相位 (包括半波损失)
        phase = (k-1) * beta + np.pi
        
        # 添加到总振幅
        Er_s += A_ks * cmath.exp(1j * phase)
        Er_p += A_kp * cmath.exp(1j * phase)
    
    # 计算总反射光强
    I_r = abs(Er_s)**2 + abs(Er_p)**2
    
    # 入射光强 (归一化为1)
    I_in = 1.0
    
    # 反射率
    R = I_r / I_in
    
    return R

# 从两个相邻极大值点求解n2和d
def solve_n2_d_from_peaks(peak1, peak2):
    """
    从两个相邻反射率极大值点求解n2和d
    
    参数:
    peak1: 第一个极大值点 (wave_number, reflectivity)
    peak2: 第二个极大值点 (wave_number, reflectivity)
    
    返回:
    (n2, d)
    """
    v1, R1 = peak1
    v2, R2 = peak2
    
    # 计算波数差
    delta_v = abs(v2 - v1)
    
    # 计算m值
    m = round((v1 + v2) / (2 * delta_v))
    
    # 定义求解n2的方程
    def equation(n2):
        # 计算对应的d值
        cos_theta2 = math.sqrt(1 - (n1 * math.sin(theta1) / n2)**2)
        d = (2 * m - 1) / (4 * n2 * cos_theta2 * v1)
        
        # 计算该点的反射率
        lamb = 1 / v1
        R_calculated = calculate_reflectivity(n2, d, lamb)
        
        # 返回反射率差异
        return R_calculated - R1
    
    # 求解n2 (假设n2在2.0到4.0之间)
    n2_solution = fsolve(equation, 3.0)[0]
    
    # 计算d
    cos_theta2 = math.sqrt(1 - (n1 * math.sin(theta1) / n2_solution)**2)
    d_solution = 1 / (2 * delta_v * math.sqrt(n2_solution**2 - n1**2 * math.sin(theta1)**2))
    
    return n2_solution, d_solution

# 示例使用
if __name__ == "__main__":
    # 示例数据 (两个相邻反射率极大值点)
    # 格式: (波数 cm⁻¹, 反射率)
    peak1 = (450.826657 ,0.75981021)  # 第一个极大值点
    peak2 = (752.409146, 0.70143102)  # 第二个极大值点
    
    # 求解n2和d
    n2, d = solve_n2_d_from_peaks(peak1, peak2)
    
    print(f"求解结果:")
    print(f"外延层折射率 n2 = {n2:.4f}")
    print(f"外延层厚度 d = {d:.6f} μm")
    
    # 验证求解结果
    print("\n验证求解结果:")
    v1, R1 = peak1
    lamb1 = 1 / v1
    R_calculated1 = calculate_reflectivity(n2, d, lamb1)
    print(f"在波数 {v1} cm⁻¹ 处:")
    print(f"  实验反射率: {R1:.4f}, 计算反射率: {R_calculated1:.4f}, 差异: {abs(R1 - R_calculated1):.6f}")
    
    v2, R2 = peak2
    lamb2 = 1 / v2
    R_calculated2 = calculate_reflectivity(n2, d, lamb2)
    print(f"在波数 {v2} cm⁻¹ 处:")
    print(f"  实验反射率: {R2:.4f}, 计算反射率: {R_calculated2:.4f}, 差异: {abs(R2 - R_calculated2):.6f}")