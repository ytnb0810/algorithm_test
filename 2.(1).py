import pulp
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import random
import warnings

warnings.filterwarnings('ignore')

# 读取四个表格的数据
file_1 = '23年种植情况.xlsx'
file_2 = '作物总销售量及销售价格.xlsx'
file_3 = '作物可种地块类型.xlsx'
file_4 = '地块类型和可种植面积.xlsx'

# 加载表格
df1 = pd.read_excel(file_1)
df2 = pd.read_excel(file_2)
df3 = pd.read_excel(file_3)
df4 = pd.read_excel(file_4)

# 定义随机选择作物的数量，减少计算量
num_crops_to_select = 9  # 可以根据需求调整
# 地块信息
land_types = df4['地块名称'].unique().tolist()  # 所有地块的列表
land_types_second_season = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                            'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14',
                            'E15', 'E16',
                            'F1', 'F2', 'F3', 'F4']


# 定义随机生成函数
def random_range(value, percent):
    return value * (1 + np.random.uniform(-percent, percent))


# 生成销售量、亩产量、种植成本和销售价格的随机值
def generate_random_parameters(crop, year, crop_type):
    yield_data = df1[(df1['作物名称_x'] == crop)]['亩产量/斤'].values[0]
    price_data = df2[df2['作物名称'] == crop]['销售单价/(元/斤)'].values[0]
    cost_data = df1[(df1['作物名称_x'] == crop)]['种植成本/(元/亩)'].values[0]
    sales_volume_2023 = df2[df2['作物名称'] == crop]['销售量/斤'].values[0]

    if crop_type == '粮食':
        sales_volume = sales_volume_2023 * (1 + np.random.uniform(0.05, 0.1)) ** (year - 2023)
    else:
        sales_volume = sales_volume_2023 * (1 + np.random.uniform(-0.05, 0.05)) ** (year - 2023)

    yield_per_acre = random_range(yield_data, 0.1)
    cost_per_acre = cost_data * (1 + 0.05) ** (year - 2023)

    if crop_type == '粮食':
        price_per_unit = price_data
    elif crop_type == '蔬菜':
        price_per_unit = price_data * (1 + 0.05) ** (year - 2023)
    else:
        price_decrease = 0.01 * np.random.uniform(1, 5) if crop != '羊肚菌' else 0.05
        price_per_unit = price_data * (1 - price_decrease) ** (year - 2023)

    return sales_volume, yield_per_acre, cost_per_acre, price_per_unit


# 获取适合某一季的作物
def get_suitable_crops(season):
    if season == '第一季':
        suitable_crops = df3[(df3['水浇地第一季'] == 1) |
                             (df3['普通大棚第一季'] == 1) |
                             (df3['智慧大棚第一季'] == 1) |
                             (df3['平旱地'] == 1) |
                             (df3['梯田'] == 1) |
                             (df3['山坡地'] == 1)]['作物名称'].tolist()
    else:
        suitable_crops = df3[(df3['水浇地第二季'] == 1) |
                             (df3['普通大棚第二季'] == 1) |
                             (df3['智慧大棚第二季'] == 1)]['作物名称'].tolist()
    return suitable_crops


# 过滤适合第一季和第二季的作物
suitable_crops_first_season = get_suitable_crops('第一季')
suitable_crops_second_season = get_suitable_crops('第二季')


# 根据地块类型选择作物
def create_land_crop_mapping(land_types, crops, season):
    land_crop_mapping = {}
    for land in land_types:
        land_type = df4[df4['地块名称'] == land]['地块类型'].values[0]

        # 根据地块类型和季节选择适合的作物
        if season == '第一季':
            if land_type == '平旱地':
                suitable_for_land = df3[df3['平旱地'] == 1]['作物名称'].tolist()
            elif land_type == '梯田':
                suitable_for_land = df3[df3['梯田'] == 1]['作物名称'].tolist()
            elif land_type == '山坡地':
                suitable_for_land = df3[df3['山坡地'] == 1]['作物名称'].tolist()
            elif land_type == '水浇地':
                suitable_for_land = df3[df3['水浇地第一季'] == 1]['作物名称'].tolist()
            elif land_type == '普通大棚':
                suitable_for_land = df3[df3['普通大棚第一季'] == 1]['作物名称'].tolist()
            elif land_type == '智慧大棚':
                suitable_for_land = df3[df3['智慧大棚第一季'] == 1]['作物名称'].tolist()
        elif season == '第二季':
            if land_type == '水浇地':
                suitable_for_land = df3[df3['水浇地第二季'] == 1]['作物名称'].tolist()
            elif land_type == '普通大棚':
                suitable_for_land = df3[df3['普通大棚第二季'] == 1]['作物名称'].tolist()
            elif land_type == '智慧大棚':
                suitable_for_land = df3[df3['智慧大棚第二季'] == 1]['作物名称'].tolist()

        available_crops = [crop for crop in crops if crop in suitable_for_land]

        # 随机选择作物
        if len(available_crops) > num_crops_to_select:
            selected_crops = random.sample(available_crops, num_crops_to_select)
        else:
            selected_crops = available_crops

        land_crop_mapping[land] = selected_crops

    return land_crop_mapping


# 处理水稻种植后的限制：水稻种植后的水浇地不能种第二季作物
def adjust_for_rice(land_crop_mapping, crops):
    for land, selected_crops in land_crop_mapping.items():
        if '水稻' in selected_crops and df4[df4['地块名称'] == land]['地块类型'].values[0] == '水浇地':
            land_crop_mapping[land] = ['水稻']  # 确保水稻种完后当年不再种其他作物
    return land_crop_mapping


# 优化函数
def optimize_land_crop(land_crop_mapping, crops, season, year):
    decision_vars = {}
    objective_coeffs = []
    A_ub = []
    b_ub = []

    for land, selected_crops in land_crop_mapping.items():
        for crop in selected_crops:
            decision_vars[(crop, land)] = 0
            crop_type = df3[df3['作物名称'] == crop]['作物类型'].values[0]
            sales_volume, yield_per_acre, cost_per_acre, price_per_unit = generate_random_parameters(crop, year,
                                                                                                     crop_type)

            # 计算净收入和超产部分的滞销收入
            sales_volume_2023 = df2[df2['作物名称'] == crop]['销售量/斤'].values[0]
            excess_volume = sales_volume - sales_volume_2023
            if excess_volume > 0:
                discounted_price = 0.5 * df2[df2['作物名称'] == crop]['销售单价/(元/斤)'].values[0]
                net_revenue = (sales_volume_2023 * price_per_unit - cost_per_acre) + (
                            excess_volume * discounted_price - cost_per_acre)
            else:
                net_revenue = sales_volume * price_per_unit - cost_per_acre

            objective_coeffs.append(-net_revenue)

    objective_coeffs = np.array(objective_coeffs)

    # 约束条件
    for land in land_crop_mapping.keys():
        constraint = np.zeros(len(decision_vars))
        for i, (crop, land_name) in enumerate(decision_vars.keys()):
            if land_name == land:
                constraint[i] = 1
        A_ub.append(constraint)
        b_ub.append(df4[df4['地块名称'] == land]['地块面积/亩'].values[0])

    for land in land_crop_mapping.keys():
        min_area = 0.1 * df4[df4['地块名称'] == land]['地块面积/亩'].values[0]
        for crop in land_crop_mapping[land]:
            constraint = np.zeros(len(decision_vars))
            for i, (crop_name, land_name) in enumerate(decision_vars.keys()):
                if crop_name == crop and land_name == land:
                    constraint[i] = -1
            A_ub.append(constraint)
            b_ub.append(-min_area)

    # 约束条件：三年内至少种植一次豆类作物
    beans_crops = df3[df3['作物类型'].str.contains('粮食（豆类）')]['作物名称'].tolist()
    for land in land_crop_mapping.keys():
        constraint = np.zeros(len(decision_vars))
        for i, (crop, land_name) in enumerate(decision_vars.keys()):
            if land_name == land and crop in beans_crops:
                constraint[i] = -1
        A_ub.append(constraint)
        b_ub.append(0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # 执行线性规划优化
    result = linprog(c=objective_coeffs, A_ub=A_ub, b_ub=b_ub, method='highs')

    # 构建优化结果的表格输出
    if result.success:
        optimal_areas = result.x
        solution = {}

        for i, (crop, land) in enumerate(decision_vars.keys()):
            if land not in solution:
                solution[land] = {}
            solution[land][crop] = optimal_areas[i]

        all_crops = sorted(set(df2['作物名称'].to_list()))

        # 创建结果表格
        results = pd.DataFrame(columns=['年', '季别', '地块名'] + all_crops)

        for land, crop_areas in solution.items():
            season_data = {'年': year, '季别': season, '地块名': land}
            for crop in all_crops:
                season_data[crop] = crop_areas.get(crop, 0)

            results = pd.concat([results, pd.DataFrame([season_data])], ignore_index=True)

        return results
    else:
        print(f"{year}年{season}优化失败，无法生成结果表格。")
        return None


# 迭代计算 2024~2030 年的结果
years = list(range(2024, 2031))
all_results = []

for year in years:
    # 每年随机选择作物
    land_crop_mapping_first_season = create_land_crop_mapping(land_types, suitable_crops_first_season, '第一季')
    land_crop_mapping_second_season = create_land_crop_mapping(land_types_second_season, suitable_crops_second_season,
                                                               '第二季')

    # 调整水稻种植后的限制
    land_crop_mapping_second_season = adjust_for_rice(land_crop_mapping_second_season, suitable_crops_second_season)

    # 第一季
    results_first_season = optimize_land_crop(land_crop_mapping_first_season, suitable_crops_first_season, '第一季',
                                              year)
    # 第二季
    results_second_season = optimize_land_crop(land_crop_mapping_second_season, suitable_crops_second_season, '第二季',
                                               year)

    # 合并两季结果
    if results_first_season is not None and results_second_season is not None:
        combined_results = pd.concat([results_first_season, results_second_season], ignore_index=True)
        all_results.append(combined_results)

# 保存所有结果到 Excel 文件
if all_results:
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_excel('2024至2030年农作物种植方案  预存 .xlsx', index=None)

# 显示生成的结果的前几行
final_results.head()

import pandas as pd

# 定义新的列名
new_columns = [
    '季别', '地块名', '黄豆', '黑豆', '红豆', '绿豆', '爬豆', '小麦', '玉米', '谷子', '高粱', '黍子', '荞麦', '南瓜',
    '红薯', '莜麦', '大麦', '水稻', '豇豆', '刀豆', '芸豆', '土豆', '西红柿', '茄子', '菠菜', '青椒', '菜花', '包菜',
    '油麦菜', '小青菜', '黄瓜', '生菜', '辣椒', '空心菜', '黄心菜', '芹菜', '大白菜', '白萝卜', '红萝卜', '榆黄菇',
    '香菇', '白灵菇', '羊肚菌'
]

# 创建一个 Pandas Excel writer 对象
with pd.ExcelWriter('2024至2030年农作物种植方案Q2 需要整理成最终.xlsx', engine='xlsxwriter') as writer:
    for year in range(2024, 2031):
        year_data = final_results[final_results['年'] == year]
        year_data = year_data.drop(columns=['年'])

        year_data = year_data.rename(columns=lambda x: x if x in new_columns else x)
        year_data = year_data.reindex(columns=new_columns, fill_value=0)
        year_data.to_excel(writer, sheet_name=str(year), index=False)

