
from itertools import combinations
import pickle
from pathlib import Path
import  sys
import os 


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#import all dataset needed     
from .generate_objects import *
from .generate_objects import load_obj_data

g_obj, ingredients_obj, herb_obj, herb_info, fangji, disease_obj, herb_distance_obj = load_obj_data()

# with open((Path(__file__).parent / "network_dictionary.pkl"), 'rb') as file:
#     network_dictionary_all = pickle.load(file)
    
# g_obj = network_dictionary_all['g_obj']
# ingredients_obj = network_dictionary_all['ingredients_obj']
# herb_obj = network_dictionary_all['herb_obj']
# herb_distance_obj = network_dictionary_all['herb_distance_obj']
# fangji = network_dictionary_all['fangji']
# herb_info = network_dictionary_all['herb_info']
# disease_obj = network_dictionary_all['diseease_obj']
    
from .disease import *
import matplotlib.pyplot as plt
import networkx as nx
g_obj.get_degree_binning(1001)
import pandas as pd

# # 导入其他模块
# from .construct_network import *
# from .proximity_key import *
# from .disease import *
# from .herb_distance_generation import *
# from .herb_herb_pairs import *
# from .herb_ingre_tar import *

# # 使用数据加载器
# from .data_loader import (
#     get_g_obj,
#     get_herb_obj,
#     get_ingredients_obj,
#     get_herb_distance_obj,
#     get_fangji,
#     get_herb_info,
#     get_disease_obj
# )

# # 提供数据对象的快捷访问
# g_obj = get_g_obj()
# herb_obj = get_herb_obj()
# ingredients_obj = get_ingredients_obj()
# herb_distance_obj = get_herb_distance_obj()
# fangji = get_fangji()
# herb_info = get_herb_info()
# disease_obj = get_disease_obj()

# this is used for changing data type
def expand_list(df, list_column, new_column):
    lens_of_lists = df[list_column].apply(len)
    origin_rows = range(df.shape[0])
    destination_rows = np.repeat(origin_rows, lens_of_lists)
    non_list_cols = (
      [idx for idx, col in enumerate(df.columns)
       if col != list_column]
    )
    expanded_df = df.iloc[destination_rows, non_list_cols].copy()
    expanded_df[new_column] = (
      [item for items in df[list_column] for item in items]
      )
    expanded_df.reset_index(inplace=True, drop=True)
    return expanded_df


class Herb_Pair_network:
    def __init__(self, herb_distance_obj, herb_from_name, herb_to_name, ingre_method, herb_method, herb_info):
        self.herb_from_name = herb_from_name
        self.herb_to_name = herb_to_name
        self.herb_distance_obj = herb_distance_obj
        self.herb_from = herb_info.herb_pinyin_dic[self.herb_from_name]
        self.herb_to = herb_info.herb_pinyin_dic[self.herb_to_name]
        self.distance_network = herb_distance_obj.herb_herb_dis_all(self.herb_from, self.herb_to)
        self.ingre_method = ingre_method
        self.herb_method = herb_method
        self.ingre_distance_dict_list = self.distance_network[self.ingre_method]['two_level']['length_dict']
        self.herb_level_distance = self.get_herb_level_distance()
        self.pd_ingre_pairs_dis = self.get_ingre_distance_pd()
        self.herb_ingre_id_name = self.get_herb_ingre_id_name_dict()
        self.pd_herb_ingre = self.get_herb_ingre_pd()
        self.center_ingredients = self.get_center_ingredients()
        self.ingre_z_dict = defaultdict()
        self.herb_z_dict = defaultdict()
        self.herb_ingre_z_dict = defaultdict()

    def get_herb_level_distance(self):
        return self.distance_network[self.ingre_method]['two_level']['distances'][self.herb_method]

    def name_find(self, ingre_id):
        return ingredients_obj.ingredients_info(ingre_id)['name']

    def name_trans_herb(self, herb_id):
        return herb_info.pinyin_herbid_dic[herb_id]

    def get_ingre_distance_pd(self):
        pd_ingredients = pd.concat([pd.DataFrame.from_dict(i, orient='index').stack().reset_index()
                                    for i in self.ingre_distance_dict_list])
        pd_ingredients.columns = ['node_from', 'node_to', 'distance']
        pd_ingredients['node_from_name'] = pd_ingredients['node_from'].apply(self.name_find)
        pd_ingredients['node_to_name'] = pd_ingredients['node_to'].apply(self.name_find)
        return pd_ingredients

    def get_herb_ingre_id_name_dict(self):
        return {self.herb_from: {ingre: self.name_find(ingre) for ingre in self.ingre_distance_dict_list[2].keys()},
                self.herb_to: {ingre: self.name_find(ingre) for ingre in self.ingre_distance_dict_list[3].keys()}}

    def get_herb_ingre_pd(self):
        pd_herb_ingre = pd.DataFrame.from_dict({k: v.keys() for k, v in self.herb_ingre_id_name.items()},
                                               orient='index').stack().reset_index()
        pd_herb_ingre.columns = ['node_from', 'index_add', 'node_to']
        pd_herb_ingre = pd_herb_ingre.drop(['index_add'], axis=1)
        pd_herb_ingre['node_from_name'] = pd_herb_ingre['node_from'].apply(self.name_trans_herb)
        pd_herb_ingre['node_to_name'] = pd_herb_ingre['node_to'].apply(self.name_find)

        return pd_herb_ingre

    def get_center_ingredients(self):
        ingredients_from = get_center_one(self.ingre_distance_dict_list[2].keys(), self.ingre_distance_dict_list[2])
        ingredients_to = get_center_one(self.ingre_distance_dict_list[3].keys(), self.ingre_distance_dict_list[3])
        return {self.herb_from_name: {ingre: self.name_find(ingre) for ingre in ingredients_from},
                self.herb_to_name: {ingre: self.name_find(ingre) for ingre in ingredients_to}}

    # def get_disease_herb_ingre_z(self, disease_obj, disease, herb, distance_method, herb_ingre_dict, ingre_tar_dict,
    #                              random_time, seed):
    #     ingre_disease_dict = disease_obj.cal_disease_herb_ingre_z_score(self, disease, herb, distance_method,
    #                                                                     herb_ingre_dict, ingre_tar_dict,
    #                                                                     random_time, seed)
    #     ingre_disease_pd = pd.DataFrame.from_dict(ingre_disease_dict, orient='index',
    #                                               columns=['d', 'z', 'm', 's', 'p_val'])
    #     ingre_disease_pd['herb'] = self.name_trans_herb(herb)
    #     ingre_disease_pd['herb_id'] = herb
    #     ingre_disease_pd['ingre_id'] = ingre_disease_pd.index
    #     ingre_disease_pd['ingre_name'] = ingre_disease_pd['ingre_id'].apply(self.name_find)
    #
    #     return ingre_disease_dict, ingre_disease_pd

    def get_disease_herb_ingre_z(self, disease_from, random_time, seed):
        herb_ingre_disease_z = defaultdict()
        herb_ingre_disease_z_from = self.herb_distance_obj.cal_herb_ingre_disease(disease_from, self.herb_from, 'closest',random_time, seed)
        herb_ingre_disease_z_to = self.herb_distance_obj.cal_herb_ingre_disease(disease_from, self.herb_to, 'closest',random_time, seed)
        herb_ingre_disease_z[self.herb_from] = herb_ingre_disease_z_from
        herb_ingre_disease_z[self.herb_to] = herb_ingre_disease_z_to
        self.herb_ingre_disease_z = herb_ingre_disease_z
        return herb_ingre_disease_z

    def get_disease_herb_z(self, disease_from, random_time, seed):
        self.herb_disease_z = self.herb_distance_obj.cal_herb_disease(disease_from, self.herb_from, 'closest',
                                                                      random_time, seed)
        herb_disease_z = defaultdict()
        herb_disease_z_from = self.herb_distance_obj.cal_herb_disease(disease_from, self.herb_from, 'closest',
                                                                      random_time, seed)
        herb_disease_z_to = self.herb_distance_obj.cal_herb_disease(disease_from, self.herb_to, 'closest',
                                                                      random_time, seed)
        herb_disease_z[self.herb_from] = herb_disease_z_from
        herb_disease_z[self.herb_to] = herb_disease_z_to
        self.herb_disease_z = herb_disease_z
        return herb_disease_z

def cal_combination_disease(herb_list, herb_distance_obj, disease_from_list, random_time=100):
    # prepare herb-disease,
    herb_disease_list = []
    herb_disease_ingre_list = []
    for herb_from_name in herb_list:
        herb_from = herb_info.herb_pinyin_dic.get(herb_from_name)
        print(herb_from)
        for disease_from in disease_from_list:
            disease_ingre_pd = herb_distance_obj.cal_herb_ingre_disease(disease_from, herb_from, 'closest', random_time, 333)
            disease_herb_pd = herb_distance_obj.cal_herb_disease(disease_from, herb_from, 'closest',random_time, 333)
            herb_disease_list.append(disease_herb_pd)
            herb_disease_ingre_list.append(disease_ingre_pd)
    herb_disease_pd = pd.concat(herb_disease_list, axis=0)
    herb_disease_ingre_pd = pd.concat(herb_disease_ingre_list, axis=0)

    return herb_disease_pd, herb_disease_ingre_pd


def cal_combination_herb_paired(herb_list, herb_distance_obj):
    dis_list = []
    center_dict = defaultdict()
    herb_distance_pd = []
    herb_ingre_center_pd = pd.DataFrame(columns=['pairs', 'herb', 'center_ingredient'])
    herb_disease_list = []
    # prepare herb-disease,

    for herb_pairs in list(combinations(herb_list, 2)):
        print(herb_pairs)
        herb1, herb2 = herb_pairs[0], herb_pairs[1]
        network_closest = Herb_Pair_network(herb_distance_obj, herb1, herb2, 'closest', 'closest', herb_info)

        huang_gan_dis_pd = network_closest.pd_ingre_pairs_dis
        huang_gan_dis_pd['herb1_name'] = herb1
        huang_gan_dis_pd['herb2_name'] = herb2
        dis_list.append(huang_gan_dis_pd)

        distance = network_closest.herb_level_distance
        herb1_id = herb_info.herb_pinyin_dic[herb1]
        herb2_id = herb_info.herb_pinyin_dic[herb2]

        herb_distance_pd.append([herb1, herb1_id, herb2, herb2_id, distance])
        network_closest.center_ingredients.update({'distance': distance})
        center_dict[herb1 + herb2] = network_closest.center_ingredients

    # center_pd = prepare_center_distance_list(center_dict)
    herb_ingre_dis_pd = pd.DataFrame(herb_distance_pd, columns=['herb id',
                                                                'herb1_name',
                                                                'herb2',
                                                                'herb2_name',
                                                                'distance'])


    ingre_ingre_dis_pd = pd.concat(dis_list)

    # prepare herb ID -ingredeint id
    herb_id_list = [herb_info.herb_pinyin_dic.get(h) for h in herb_list]
    herb_ingre_dict_used = {k: v for k, v in herb_distance_obj.Herb.herb_ingre_dict.items() if k in herb_id_list}
    herb_ingre_id_pairs = pd.DataFrame.from_dict(
        {'herb_id': herb_ingre_dict_used.keys(), 'ingredient_id': herb_ingre_dict_used.values()})
    herb_ingre_id_pairs = expand_list(herb_ingre_id_pairs, 'ingredient_id', 'ingredient_id')
    herb_ingre_id_pairs['herb_name'] = herb_ingre_id_pairs['herb_id'].apply(lambda x:herb_info.pinyin_herbid_dic[x])
    herb_ingre_id_pairs['ingredient_name'] = herb_ingre_id_pairs['ingredient_id'].apply(lambda x:ingredients_obj.ingre_id_name_dict_value[x])
    return herb_ingre_dis_pd, ingre_ingre_dis_pd, herb_ingre_id_pairs



#The final file generation and processing.
def cal_combination_distance(herb_info, herb_list):
    herb_all = herb_info.data
    is_in_column = herb_all['Pinyin Name'].isin(herb_list)
    rows_containing_elements = herb_all[is_in_column]
    herb_list = rows_containing_elements['Pinyin Name'].tolist()
    herb_herb_dis_pd, ingre_ingre_dis_pd, herb_ingre_id_pairs = cal_combination_herb_paired(herb_list, herb_distance_obj)
    herb_herb_dis = herb_herb_dis_pd.iloc[:, [0, 2, 4]]
    herb_herb_dis.columns = ['herb1', 'herb2', 'distance']
    herb_herb_dis = herb_herb_dis.drop_duplicates().dropna()
    ingre_ingre_dis=ingre_ingre_dis_pd.iloc[:,[3,4,2]]
    ingre_ingre_dis['Combination'] = ingre_ingre_dis[['node_from_name', 'node_to_name']].apply(lambda x: ''.join(sorted(x)), axis=1)
    ingre_ingre_dis = ingre_ingre_dis.drop_duplicates(subset=['Combination', 'distance'])
    ingre_ingre_dis =ingre_ingre_dis.drop(columns=['Combination'])
    ingre_ingre_dis =ingre_ingre_dis.dropna().drop_duplicates()
    ingre_ingre_dis.columns=['ingre1','ingre2','distance']
    herb_ingre_pairs = herb_ingre_id_pairs.iloc[:, [2, 3]]
    herb_ingre_pairs=herb_ingre_pairs.dropna().drop_duplicates()
    herb_ingre_pairs.columns = ['herb', 'ingredient']
    return herb_list, herb_herb_dis, ingre_ingre_dis, herb_ingre_pairs



#Taking Suhuang Zhike Capsules as an example
def main():
    combination_herb_list=['MA HUANG','ZI SU YE','ZI SU ZI','CHAN TUI','QIAN HU','NIU BANG ZI','WU WEI ZI','DI LONG','PI PA YE']
    disease_list = ['Cough Variant Asthma']
    herb_list, herb_herb_dis, ingre_ingre_dis, herb_ingre_pairs = cal_combination_distance(herb_info,combination_herb_list)
    herb_disease_pd, herb_disease_ingre_pd = cal_combination_disease(herb_list, herb_distance_obj, disease_obj,
                                                                disease_list)
    ##save_file
    # herb_herb_dis.to_csv('result/SH_herb_herb_distance.csv', index=None)
    # ingre_ingre_dis.to_csv('result/SH_ingre_ingre_distance.csv', index=None)
    # herb_ingre_pairs.to_csv('result/SH_herb_ingredient_pairs.csv', index=None)
    # herb_disease_pd.to_csv('result/SH_herb_disease_pd.csv', index=None)
    # herb_disease_ingre_pd.to_csv('result/SH_herb_disease_ingre_pd.csv', index=None)




