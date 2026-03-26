import numpy as np
import math
from scipy.optimize import fsolve

# 常量定义
n1 = 1.0  # 空气折射率
theta1_deg = 10.0  # 入射角（度）
theta1 = math.radians(theta1_deg)  # 转换为弧度

# 硅晶衬底折射率计算函数 (Sellmeier公式)
def sellmeier_si(wavelength_um):
    """根据Sellmeier公式计算硅的折射率"""
    term1 = 11.67316
    term2 = 1 / (wavelength_um ** 2)
    term3 = 0.004482633 / (wavelength_um ** 2 - 1.108205**2)
    return math.sqrt(term1 + term2 + term3)

# 振幅反射率和透射率计算函数
def calc_amplitudes(n2, lambda_nm):
    """
    计算所有需要的振幅反射率和透射率
    n2: 外延层折射率
    lambda_nm: 波长（纳米）
    """
    lambda_um = lambda_nm / 1000.0  # 转换为微米
    
    # 使用Sellmeier公式计算硅衬底折射率（不区分偏振）
    n3 = sellmeier_si(lambda_um)
    
    # 计算外延层中的折射角
    sin_theta1 = math.sin(theta1)
    cos_theta1 = math.cos(theta1)
    sin_theta2 = n1 * sin_theta1 / n2
    cos_theta2 = math.sqrt(1 - sin_theta2**2) if n2 > n1 * sin_theta1 else 0
    
    # 公共表达式
    sqrt_expr = math.sqrt(n2**2 - n1**2 * sin_theta1**2)
    
    # 计算各振幅反射率和透射率
    rs12 = (n1 * cos_theta1 - sqrt_expr) / (n1 * cos_theta1 + sqrt_expr)
    ts12 = (2 * n1 * cos_theta1) / (n1 * cos_theta1 + sqrt_expr)
    
    rp12 = (n2**2 * cos_theta1 - n1 * sqrt_expr) / (n2**2 * cos_theta1 + n1 * sqrt_expr)
    tp12 = (2 * n1 * cos_theta1) / (n2**2 * cos_theta1 / n1 + sqrt_expr)  # 简化后的等效公式
    
    # 使用相同的n3值计算s和p偏振
    rs23 = (sqrt_expr - math.sqrt(n3**2 - n1**2 * sin_theta1**2)) / (sqrt_expr + math.sqrt(n3**2 - n1**2 * sin_theta1**2))
    rp23 = (math.sqrt(n3**2 - n1**2 * sin_theta1**2) - sqrt_expr) / (math.sqrt(n3**2 - n1**2 * sin_theta1**2) + sqrt_expr)
    
    rs21 = (sqrt_expr - n1 * cos_theta1) / (sqrt_expr + n1 * cos_theta1)
    rp21 = (n1 * cos_theta1 - sqrt_expr) / (n1 * cos_theta1 + sqrt_expr)
    
    ts21 = (2 * sqrt_expr) / (sqrt_expr + n1 * cos_theta1)
    tp21 = (2 * n2 * cos_theta2) / (n1 * cos_theta2 + n2 * cos_theta1)  # 使用角度关系
    
    return {
        'rs12': rs12, 'ts12': ts12, 'rp12': rp12, 'tp12': tp12,
        'rs23': rs23, 'rp23': rp23, 'rs21': rs21, 'rp21': rp21,
        'ts21': ts21, 'tp21': tp21, 'cos_theta2': cos_theta2
    }

# 多光束干涉反射率计算
def calc_reflectivity(n2, d, lambda_nm, max_reflections=5):
    """
    计算多光束干涉的总反射率
    n2: 外延层折射率
    d: 外延层厚度（米）
    lambda_nm: 波长（纳米）
    max_reflections: 考虑的最大反射光次数
    """
    amps = calc_amplitudes(n2, lambda_nm)
    lambda_m = lambda_nm * 1e-9  # 转换为米
    
    # 相位计算相关参数
    phase_factor = 4 * math.pi * n2 * d * amps['cos_theta2'] / lambda_m
    
    # 初始化振幅和相位列表
    as_list, ap_list = [], []  # s和p偏振的振幅
    phases = []  # 相位
    
    # 第一束反射光
    as_list.append(amps['rs12'])
    ap_list.append(amps['rp12'])
    phases.append(0)  # 基准相位
    
    # 后续反射光
    for k in range(2, max_reflections + 1):
        # 计算振幅
        as_k = amps['ts12'] * amps['rs23'] * (amps['rs21'] * amps['rs23'])**(k-2) * amps['ts21']
        ap_k = amps['tp12'] * amps['rp23'] * (amps['rp21'] * amps['rp23'])**(k-2) * amps['tp21']
        
        as_list.append(as_k)
        ap_list.append(ap_k)
        
        # 计算相位：第k束光比第1束光多的相位
        phase_k = phase_factor * (k-1) + math.pi
        phases.append(phase_k)
    
    # 将振幅转换为复数（考虑相位）
    as_complex = [amp * (math.cos(phase) + 1j * math.sin(phase)) 
                  for amp, phase in zip(as_list, phases)]
    ap_complex = [amp * (math.cos(phase) + 1j * math.sin(phase)) 
                  for amp, phase in zip(ap_list, phases)]
    
    # 合成总振幅
    as_total = np.sum(as_complex)
    ap_total = np.sum(ap_complex)
    
    # 计算总反射振幅和反射率
    a_total = math.sqrt(abs(as_total)**2 + abs(ap_total)**2)
    r = a_total**2 / 2  # 反射率（入射光总振幅为√2）
    
    return r

# 求解n2的函数（用于数值优化）
def solve_n2(n2, lambda_nm, delta_nu, r_target):
    """
    用于fsolve的目标函数，求解n2
    n2: 待求解的外延层折射率
    lambda_nm: 波长（纳米）
    delta_nu: 相邻极大值的波数差（cm^{-1}）
    r_target: 目标反射率
    """
    # 计算厚度d（单位：米）
    d = 1 / (2 * delta_nu * 100 * math.sqrt(n2**2 - n1**2 * math.sin(theta1)**2))
    
    # 计算反射率
    r_calculated = calc_reflectivity(n2, d, lambda_nm)
    
    # 返回误差
    return r_calculated - r_target

# 计算单个厚度d的函数
def calculate_d_for_data(nu1, r1, nu2):
    """
    使用一组相邻极大值点数据计算厚度d
    nu1: 第一个极大值点的波数 (cm^{-1})
    r1: 第一个极大值点的反射率
    nu2: 相邻的第二个极大值点的波数 (cm^{-1})
    """
    # 计算波数差
    delta_nu = abs(nu2 - nu1)  # cm^{-1}
    
    # 计算波长（纳米）
    lambda1_nm = 1e7 / nu1  # 波数转换为波长（nm）
    
    # 使用第一组数据求解n2
    # 初始猜测值（硅外延层折射率通常在3.4-3.6之间）
    n2_initial_guess = 3.5
    
    # 数值求解n2
    n2_solution = fsolve(
        lambda n2: solve_n2(n2[0], lambda1_nm, delta_nu, r1),
        n2_initial_guess
    )[0]
    
    # 计算厚度d（单位：米）
    d_solution = 1 / (2 * delta_nu * 100 * math.sqrt(n2_solution**2 - n1**2 * math.sin(theta1)**2))
    
    return d_solution, n2_solution

# 主函数：使用多组数据求解d并求平均值
def main():
    # 示例数据（需要替换为实际数据）
    # 每组数据包含两个相邻极大值点的波数和第一个点的反射率
    data_groups = [
        # 格式: (nu1, r1, nu2)
        # 第一组相邻极大值点
        (435.9196, 0.876847, 767.7273),
        # 第二组相邻极大值点
        (767.7273, 0.785707, 1110.277),
        # 第三组相邻极大值点
        (1110.277, 0.609917, 1522.973),
        # 第四组相邻极大值点
        (1522.973, 0.432305, 1943.938),
        # 第五组相邻极大值点
        (1943.938, 0.363567, 2373.044),
    ]
    
    d_values = []  # 存储所有计算出的厚度
    n2_values = []  # 存储所有计算出的折射率
    
    print("开始处理数据组...")
    for i, (nu1, r1, nu2) in enumerate(data_groups):
        print(f"\n处理第 {i+1} 组数据:")
        print(f"  波数1: {nu1} cm⁻¹, 反射率1: {r1:.4f}")
        print(f"  波数2: {nu2} cm⁻¹")
        
        # 计算厚度d和折射率n2
        d, n2 = calculate_d_for_data(nu1, r1, nu2)
        
        print(f"  计算得到折射率 n2 = {n2:.4f}")
        print(f"  计算得到厚度 d = {d:.6e} 米 ({d * 1e6:.2f} 微米)")
        
        d_values.append(d)
        n2_values.append(n2)
    
    # 计算平均值
    avg_d = np.mean(d_values)
    avg_n2 = np.mean(n2_values)
    
    # 输出最终结果
    print("\n最终结果:")
    print(f"平均外延层折射率 n2 = {avg_n2:.4f}")
    print(f"平均外延层厚度 d = {avg_d:.6e} 米")
    print(f"                = {avg_d * 1e6:.2f} 微米")
    
    # 输出所有计算结果
    print("\n所有计算结果:")
    for i, (d, n2) in enumerate(zip(d_values, n2_values)):
        print(f"数据组 {i+1}: n2={n2:.4f}, d={d*1e6:.2f} μm")

if __name__ == "__main__":
    main()