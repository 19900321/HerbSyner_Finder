# cal_network_distance/data_loader.py
import pickle
from pathlib import Path
import warnings

_network_data = None

def get_network_data():
    """获取网络数据（单例模式）"""
    global _network_data
    
    if _network_data is None:
        try:
            pkl_path = Path(__file__).parent / "network_dictionary.pkl"
            with open(pkl_path, 'rb') as file:
                _network_data = pickle.load(file)
        except FileNotFoundError:
            warnings.warn("network_dictionary.pkl 未找到。请先运行 generate_objects.py 生成数据。")
            _network_data = {}
    
    return _network_data

def get_g_obj():
    return get_network_data().get('g_obj')

def get_ingredients_obj():
    return get_network_data().get('ingredients_obj')

def get_herb_obj():
    return get_network_data().get('herb_obj')

def get_herb_distance_obj():
    return get_network_data().get('herb_distance_obj')

def get_fangji():
    return get_network_data().get('fangji')

def get_herb_info():
    return get_network_data().get('herb_info')

def get_disease_obj():
    return get_network_data().get('disease_obj')