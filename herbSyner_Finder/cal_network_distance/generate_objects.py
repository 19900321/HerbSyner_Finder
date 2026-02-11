import pickle
import pandas as pd
from pathlib import Path
from .herb_ingre_tar import *
from .construct_network import *
from .herb_distance_generation import *
from .herb_herb_pairs import *
from . import disease
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_obj_data():
    g_obj = Construct_Network("source_data/PPI.sif")
    g_obj.get_degree_binning(1001)

    filename = 'source_data/stitch_database_chemical_target_sum.xlsx'
    ingredients_obj = Ingredients(filename, 700, 'hum')
    ingredients_obj.ingredients_target_dict(g_obj.G.nodes)

    filename_2 = 'source_data/herb_ingre_info.csv'
    herb_obj = Herb(filename_2)
    herb_obj.herb_ingre_dic(ingredients_obj.ingre_tar_dict)
    herb_obj.herb_ingretargets_dic(ingredients_obj.ingre_tar_dict)

    filename_3 = 'source_data/herb_info.csv'
    herb_info = Herb_Info(filename_3 )

    fangji = FangJi('source_data/prescription.txt', herb_info.herb_pinyin_dic)
    disease_file_name = 'source_data/disease_genes.csv'
    disease_obj = disease.Disease(disease_file_name, g_obj)

    herb_distance_obj = Herb_Distance(g_obj, ingredients_obj, herb_obj, disease_obj, herb_info)
    
    return g_obj, ingredients_obj, herb_obj, herb_info, fangji, disease_obj, herb_distance_obj

if __name__ == '__main__':
    #g_obj, ingredients_obj, herb_obj, herb_info, fangji, disease_obj, herb_distance_obj = load_obj_data()
    pass
    
    

