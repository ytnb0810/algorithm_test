import pandas as pd
import numpy as np
from scipy.optimize import linprog
import random
import warnings

warnings.filterwarnings('ignore')
# 第一步: 加载数据
file_1_path = '23年种植情况.xlsx'   ##已更改
file_2_path = '作物总销售量及销售价格.xlsx' ##数据没变
file_3_path = '作物可种地块类型.xlsx'  ##数据已替换
file_4_path = '地块类型和可种植面积.xlsx' #  #数据已替换

df1 = pd.read_excel(file_1_path)
df2 = pd.read_excel(file_2_path)
df3 = pd.read_excel(file_3_path)
df4 = pd.read_excel(file_4_path)

# 提取出2023年销售数据
sales_2023 = df2.set_index('作物名称')['销售量/斤'].to_dict()

# 地块信息
land_types = df4['地块名称'].unique().tolist()  # 所有地块的列表
land_types_second_season = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                            'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14',
                            'E15', 'E16',
                            'F1', 'F2', 'F3', 'F4']

# 每个地块选择作物的数量，减少计算量
num_crops_to_select = 9


# 第二步: 定义函数，分类适合第一季和第二季种植的的作物
def get_suitable_crops(season):
    suitable_crops = []
    if season == '第一季':
        suitable_crops = df3[(df3['水浇地第一季'] == 1) |
                             (df3['普通大棚第一季'] == 1) |
                             (df3['智慧大棚第一季'] == 1) |
                             (df3['平旱地'] == 1) |
                             (df3['梯田'] == 1) |
                             (df3['山坡地'] == 1)]['作物名称'].tolist()
    elif season == '第二季':
        suitable_crops = df3[(df3['水浇地第二季'] == 1) |
                             (df3['普通大棚第二季'] == 1) |
                             (df3['智慧大棚第二季'] == 1)]['作物名称'].tolist()
    return suitable_crops


suitable_crops_first_season = get_suitable_crops('第一季')
suitable_crops_second_season = get_suitable_crops('第二季')


# 第三步: 创建地块与作物的映射，根据地块类型和季别筛选作物
def create_land_crop_mapping(land_types, crops, season):
    land_crop_mapping = {}
    for land in land_types:
        # 获取地块类型
        land_type = df4[df4["地块名称"] == land]["地块类型"].values[0]

        # 根据地块类型和季别筛选适合的作物
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

        # 过滤适合该地块类型的作物
        available_crops = [crop for crop in crops if crop in suitable_for_land]

        # 随机选择作物
        if len(available_crops) > num_crops_to_select:
            selected_crops = random.sample(available_crops, num_crops_to_select)
        else:
            selected_crops = available_crops

        # 保存该地块对应的作物
        land_crop_mapping[land] = selected_crops

    return land_crop_mapping


# 第四步: 处理特殊情况：水稻种植后的水浇地不能种第二季作物
def adjust_for_rice(land_crop_mapping, crops):
    for land, selected_crops in land_crop_mapping.items():
        if '水稻' in selected_crops and df4[df4['地块名称'] == land]['地块类型'].values[0] == '水浇地':
            # 移除水浇地第二季作物
            land_crop_mapping[land] = ['水稻']  # 确保水稻种完后当年不再种其他作物
    return land_crop_mapping


# Step 5: 优化函数，最大化净收益并最小化滞销损失
def optimize_land_crop(land_crop_mapping, crops, season, year):
    decision_vars = []
    objective_coeffs = []

    for land, selected_crops in land_crop_mapping.items():
        for crop in selected_crops:
            decision_vars.append((crop, land))

            land_type = df4[df4['地块名称'] == land]['地块类型'].values[0]
            yield_data = df1[(df1['作物名称_x'] == crop) & (df1['地块类型'] == land_type)]['亩产量/斤'].values
            price_data = df2[df2['作物名称'] == crop]['销售单价/(元/斤)'].values
            cost_data = df1[(df1['作物名称_x'] == crop) & (df1['地块类型'] == land_type)]['种植成本/(元/亩)'].values

            if len(yield_data) > 0 and len(price_data) > 0 and len(cost_data) > 0:
                yield_per_acre = yield_data[0]
                price_per_unit = price_data[0]
                cost_per_acre = cost_data[0]

                # 计算净收益和滞销损失
                sales_2023_volume = sales_2023.get(crop, 0)
                net_revenue = (yield_per_acre * price_per_unit) - cost_per_acre
                excess_volume = max(0, yield_per_acre - sales_2023_volume)
                wastage_loss = excess_volume * price_per_unit

                objective_coeffs.append(net_revenue - wastage_loss)
            else:
                objective_coeffs.append(0)

    objective_coeffs = np.array(objective_coeffs) * -1

    A_ub = []
    b_ub = []

    # 约束条件1: 总面积约束
    for land in land_crop_mapping.keys():
        constraint = np.zeros(len(decision_vars))
        for i, (crop, land_name) in enumerate(decision_vars):
            if land_name == land:
                constraint[i] = 1
        A_ub.append(constraint)
        b_ub.append(df4[df4['地块名称'] == land]['地块面积/亩'].values[0])

    # 约束条件2: 最小种植面积
    for land in land_crop_mapping.keys():
        min_area = 0.1 * df4[df4['地块名称'] == land]['地块面积/亩'].values[0]
        for crop in land_crop_mapping[land]:
            constraint = np.zeros(len(decision_vars))
            for i, (crop_name, land_name) in enumerate(decision_vars):
                if crop_name == crop and land_name == land:
                    constraint[i] = -1
            A_ub.append(constraint)
            b_ub.append(-min_area)

    # 三年内至少种植一次豆类作物的约束
    beans_crops = df3[df3['作物类型'].str.contains('粮食（豆类）')]['作物名称'].tolist()
    for land in land_crop_mapping.keys():
        constraint = np.zeros(len(decision_vars))
        for i, (crop, land_name) in enumerate(decision_vars):
            if land_name == land and crop in beans_crops:
                constraint[i] = -1
        A_ub.append(constraint)
        b_ub.append(0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # 执行线性规划求解
    result = linprog(c=objective_coeffs, A_ub=A_ub, b_ub=b_ub, method='highs')

    # 生成结果表格
    if result.success:
        optimal_areas = result.x
        solution = {}

        for i, (crop, land) in enumerate(decision_vars):
            if land not in solution:
                solution[land] = {}
            solution[land][crop] = optimal_areas[i]

        # 创建结果表格
        all_crops = sorted(set(df2['作物名称'].to_list()))  # 获取所有作物列表
        results = pd.DataFrame(columns=['年', '季别', '地块名'] + all_crops)

        for land, crop_areas in solution.items():
            season_data = {'年': year, '季别': season, '地块名': land}
            for crop in all_crops:
                season_data[crop] = crop_areas.get(crop, 0)  # 如果作物不在该地块种植，则为0

            # 将当前地块的作物数据添加到结果中
            results = pd.concat([results, pd.DataFrame([season_data])], ignore_index=True)

        return results
    else:
        print(f"{year}年{season}优化失败，无法生成结果表格。")
        return None


# Step 6: 执行优化并随机每年选择不同作物
years = list(range(2024, 2031))
all_results = []

for year in years:
    # 每年随机选择作物
    land_crop_mapping_first_season = create_land_crop_mapping(land_types, suitable_crops_first_season, '第一季')
    land_crop_mapping_second_season = create_land_crop_mapping(land_types_second_season, suitable_crops_second_season,
                                                               '第二季')

    # 检查并调整水稻种植后的限制
    land_crop_mapping_second_season = adjust_for_rice(land_crop_mapping_second_season, suitable_crops_second_season)

    # 第一季
    results_first_season = optimize_land_crop(land_crop_mapping_first_season, suitable_crops_first_season, '第一季',
                                              year)
    # 第二季
    results_second_season = optimize_land_crop(land_crop_mapping_second_season, suitable_crops_second_season, '第二季',
                                               year)

    # 合并两季的结果
    if results_first_season is not None and results_second_season is not None:
        combined_results = pd.concat([results_first_season, results_second_season], ignore_index=True)
        all_results.append(combined_results)


# Step 7: 保存所有结果到 Excel 文件
if all_results:
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.to_excel('2024至2030年农作物种植方案.xlsx', index=None)

# 显示生成的结果的前几行
final_results.head()
