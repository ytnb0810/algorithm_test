import pandas as pd
import numpy as np
from scipy.optimize import linprog
import random
import warnings

warnings.filterwarnings('ignore')

# 步骤 1：加载数据
data_file_1 = '23年种植情况.xlsx'
data_file_2 = '作物总销售量及销售价格.xlsx'
data_file_3 = '作物可种地块类型.xlsx'
data_file_4 = '地块类型和可种植面积.xlsx'

dataframe_1 = pd.read_excel(data_file_1)
dataframe_2 = pd.read_excel(data_file_2)
dataframe_3 = pd.read_excel(data_file_3)
dataframe_4 = pd.read_excel(data_file_4)

# 提取 2023 年的销售数据
sales_data_2023 = dataframe_2.set_index('作物名称')['销售量/斤'].to_dict()

# 地块信息
land_names = dataframe_4['地块名称'].unique().tolist()
secondary_land_names = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
                        'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14',
                        'E15', 'E16',
                        'F1', 'F2', 'F3', 'F4']

# 每块地块选择的作物数量
crops_to_select_count = 9

# 步骤 2：定义函数以分类适合的作物
def find_suitable_crops(season_name):
    suitable_crops_list = []
    if season_name == '第一季':
        suitable_crops_list = dataframe_3[(dataframe_3['水浇地第一季'] == 1) |
                                           (dataframe_3['普通大棚第一季'] == 1) |
                                           (dataframe_3['智慧大棚第一季'] == 1) |
                                           (dataframe_3['平旱地'] == 1) |
                                           (dataframe_3['梯田'] == 1) |
                                           (dataframe_3['山坡地'] == 1)]['作物名称'].tolist()
    elif season_name == '第二季':
        suitable_crops_list = dataframe_3[(dataframe_3['水浇地第二季'] == 1) |
                                           (dataframe_3['普通大棚第二季'] == 1) |
                                           (dataframe_3['智慧大棚第二季'] == 1)]['作物名称'].tolist()
    return suitable_crops_list

first_season_crops = find_suitable_crops('第一季')
second_season_crops = find_suitable_crops('第二季')

# 步骤 3：根据类型和季节创建地块与作物的映射
def map_land_to_crops(land_names, crops_list, season_name):
    land_crop_map = {}
    for land in land_names:
        land_type = dataframe_4[dataframe_4["地块名称"] == land]["地块类型"].values[0]

        if season_name == '第一季':
            if land_type == '平旱地':
                suitable_for_this_land = dataframe_3[dataframe_3['平旱地'] == 1]['作物名称'].tolist()
            elif land_type == '梯田':
                suitable_for_this_land = dataframe_3[dataframe_3['梯田'] == 1]['作物名称'].tolist()
            elif land_type == '山坡地':
                suitable_for_this_land = dataframe_3[dataframe_3['山坡地'] == 1]['作物名称'].tolist()
            elif land_type == '水浇地':
                suitable_for_this_land = dataframe_3[dataframe_3['水浇地第一季'] == 1]['作物名称'].tolist()
            elif land_type == '普通大棚':
                suitable_for_this_land = dataframe_3[dataframe_3['普通大棚第一季'] == 1]['作物名称'].tolist()
            elif land_type == '智慧大棚':
                suitable_for_this_land = dataframe_3[dataframe_3['智慧大棚第一季'] == 1]['作物名称'].tolist()
        elif season_name == '第二季':
            if land_type == '水浇地':
                suitable_for_this_land = dataframe_3[dataframe_3['水浇地第二季'] == 1]['作物名称'].tolist()
            elif land_type == '普通大棚':
                suitable_for_this_land = dataframe_3[dataframe_3['普通大棚第二季'] == 1]['作物名称'].tolist()
            elif land_type == '智慧大棚':
                suitable_for_this_land = dataframe_3[dataframe_3['智慧大棚第二季'] == 1]['作物名称'].tolist()

        available_crops_list = [crop for crop in crops_list if crop in suitable_for_this_land]

        if len(available_crops_list) > crops_to_select_count:
            chosen_crops = random.sample(available_crops_list, crops_to_select_count)
        else:
            chosen_crops = available_crops_list

        land_crop_map[land] = chosen_crops

    return land_crop_map

# 步骤 4：处理水稻的特殊情况
def adjust_mapping_for_rice(land_crop_map, crops_list):
    for land, chosen_crops in land_crop_map.items():
        if '水稻' in chosen_crops and dataframe_4[dataframe_4['地块名称'] == land]['地块类型'].values[0] == '水浇地':
            land_crop_map[land] = ['水稻']
    return land_crop_map

# 步骤 5：优化函数以最大化净收入并最小化浪费
def optimize_crops_on_land(land_crop_map, crops_list, season_name, year_value):
    decision_variables = []
    objective_coefficients = []

    for land, chosen_crops in land_crop_map.items():
        for crop in chosen_crops:
            decision_variables.append((crop, land))

            land_type = dataframe_4[dataframe_4['地块名称'] == land]['地块类型'].values[0]
            yield_values = dataframe_1[(dataframe_1['作物名称_x'] == crop) & (dataframe_1['地块类型'] == land_type)]['亩产量/斤'].values
            price_values = dataframe_2[dataframe_2['作物名称'] == crop]['销售单价/(元/斤)'].values
            cost_values = dataframe_1[(dataframe_1['作物名称_x'] == crop) & (dataframe_1['地块类型'] == land_type)]['种植成本/(元/亩)'].values

            if yield_values.size > 0 and price_values.size > 0 and cost_values.size > 0:
                yield_per_acre_value = yield_values[0]
                price_per_unit_value = price_values[0]
                cost_per_acre_value = cost_values[0]

                sales_2023_amount = sales_data_2023.get(crop, 0)
                net_profit = (yield_per_acre_value * price_per_unit_value) - cost_per_acre_value
                excess_amount = max(0, yield_per_acre_value - sales_2023_amount)
                waste_loss = excess_amount * price_per_unit_value

                objective_coefficients.append(net_profit - waste_loss)
            else:
                objective_coefficients.append(0)

    objective_coefficients = np.array(objective_coefficients) * -1

    A_inequality = []
    b_inequality = []

    # 约束 1：总面积约束
    for land in land_crop_map.keys():
        constraint_row = np.zeros(len(decision_variables))
        for index, (crop_name, land_name) in enumerate(decision_variables):
            if land_name == land:
                constraint_row[index] = 1
        A_inequality.append(constraint_row)
        b_inequality.append(dataframe_4[dataframe_4['地块名称'] == land]['地块面积/亩'].values[0])

    # 约束 2：最小种植面积
    for land in land_crop_map.keys():
        min_area_value = 0.1 * dataframe_4[dataframe_4['地块名称'] == land]['地块面积/亩'].values[0]
        for crop in land_crop_map[land]:
            constraint_row = np.zeros(len(decision_variables))
            for index, (crop_name, land_name) in enumerate(decision_variables):
                if crop_name == crop and land_name == land:
                    constraint_row[index] = -1
            A_inequality.append(constraint_row)
            b_inequality.append(-min_area_value)

    # 三年内必须种植至少一种豆类作物
    legume_crops = dataframe_3[dataframe_3['作物类型'].str.contains('粮食（豆类）')]['作物名称'].tolist()
    for land in land_crop_map.keys():
        constraint_row = np.zeros(len(decision_variables))
        for index, (crop_name, land_name) in enumerate(decision_variables):
            if land_name == land and crop_name in legume_crops:
                constraint_row[index] = -1
        A_inequality.append(constraint_row)
        b_inequality.append(0)

    A_inequality = np.array(A_inequality)
    b_inequality = np.array(b_inequality)

    # 执行线性规划
    optimization_result = linprog(c=objective_coefficients, A_ub=A_inequality, b_ub=b_inequality, method='highs')

    # 生成结果表
    if optimization_result.success:
        optimal_area_distribution = optimization_result.x
        final_solution = {}

        for index, (crop_name, land_name) in enumerate(decision_variables):
            if land_name not in final_solution:
                final_solution[land_name] = {}
            final_solution[land_name][crop_name] = optimal_area_distribution[index]

        # 创建结果数据框
        all_crop_names = sorted(set(dataframe_2['作物名称'].to_list()))
        result_dataframe = pd.DataFrame(columns=['年', '季别', '地块名'] + all_crop_names)

        for land, crop_areas in final_solution.items():
            season_data = {'年': year_value, '季别': season_name, '地块名': land}
            for crop in all_crop_names:
                season_data[crop] = crop_areas.get(crop, 0)

            result_dataframe = pd.concat([result_dataframe, pd.DataFrame([season_data])], ignore_index=True)

        return result_dataframe
    else:
        print(f"{year_value}年{season_name}优化失败，无法生成结果表格。")
        return None


import pandas as pd

# 假设这是你的种植数据
data = {
    '种植地块': ['A4', 'A4', 'B11', 'B11', 'C3', 'C3', 'A5', 'A5'],
    '作物名称_x': ['黄豆', '黄豆', '黄豆', '黄豆', '黄豆', '黄豆', '绿豆', '绿豆'],
    '种植季次_x': ['单季', '单季', '单季', '单季', '单季', '单季', '单季', '单季']
}

df = pd.DataFrame(data)


# 定义一个函数来检查重茬
def check_no_replanting(df):
    # 创建一个空集合来跟踪已种植的作物
    planted = {}

    for index, row in df.iterrows():
        field = row['种植地块']
        crop = row['作物名称_x']

        if field not in planted:
            planted[field] = set()

        # 检查是否已经种植过该作物
        if crop in planted[field]:
            print(f"警告: 在地块 {field} 上发现重茬作物 {crop}.")
            return False

        # 添加到已种植集合中
        planted[field].add(crop)

    print("所有地块都遵循了轮作原则.")
    return True


# 调用函数检查重茬
check_no_replanting(df)
# 步骤 6：执行优化并随机选择每年的作物
year_range = list(range(2024, 2031))
results_collection = []

for year in year_range:
    # 每年随机选择作物
    first_season_land_crop_map = map_land_to_crops(land_names, first_season_crops, '第一季')
    second_season_land_crop_map = map_land_to_crops(secondary_land_names, second_season_crops, '第二季')

    # 调整水稻种植限制
    second_season_land_crop_map = adjust_mapping_for_rice(second_season_land_crop_map, second_season_crops)

    # 第一季优化
    first_season_results = optimize_crops_on_land(first_season_land_crop_map, first_season_crops, '第一季', year)
    # 第二季优化
    second_season_results = optimize_crops_on_land(second_season_land_crop_map, second_season_crops, '第二季', year)

    # 合并两个季节的结果
    if first_season_results is not None and second_season_results is not None:
        combined_results = pd.concat([first_season_results, second_season_results], ignore_index=True)
        results_collection.append(combined_results)

# 步骤 7：将所有结果保存到 Excel 文件
if results_collection:
    final_output = pd.concat(results_collection, ignore_index=True)
    final_output.to_excel('2024至2030年农作物种植方案.xlsx.xlsx', index=None)

# 显示生成结果的前几行
final_output.head()
# 定义新的列名
new_columns = [
    '季别', '地块名', '黄豆', '黑豆', '红豆', '绿豆', '爬豆', '小麦', '玉米', '谷子', '高粱', '黍子', '荞麦', '南瓜',
    '红薯', '莜麦', '大麦', '水稻', '豇豆', '刀豆', '芸豆', '土豆', '西红柿', '茄子', '菠菜', '青椒', '菜花', '包菜',
    '油麦菜', '小青菜', '黄瓜', '生菜', '辣椒', '空心菜', '黄心菜', '芹菜', '大白菜', '白萝卜', '红萝卜', '榆黄菇',
    '香菇', '白灵菇', '羊肚菌'
]

# 创建一个 Pandas Excel writer 对象
with pd.ExcelWriter('2024至2030年农作物种植方案Q1_1.xlsx', engine='xlsxwriter') as writer:
    # 遍历每个年份并将相应的数据写入到不同的工作表中
    for year in range(2024, 2031):
        # 选择对应年份的数据
        year_data = final_output[final_output['年'] == year]

        # 删除年份列
        year_data = year_data.drop(columns=['年'])

        # 确保列名顺序符合要求
        year_data = year_data.reindex(columns=new_columns, fill_value=0)

        # 将数据写入到 Excel 文件的对应工作表中
        year_data.to_excel(writer, sheet_name=str(year), index=False)