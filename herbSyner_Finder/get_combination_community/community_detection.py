import networkx as nx
import pandas as pd
import numpy as np
from communities.algorithms import louvain_method
from sklearn.preprocessing import MinMaxScaler
import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        
class Herb_Comb_Community:
    
    def __init__(self, HH, II, DH, DI, disease_name):
        self.HH, self.II, self.DH, self.DI = HH, II, DH, DI
        self.disease_name = disease_name
        self.sum_file_all = self.file_pre_processing()
        
    # 1. prepare format for communication calculation
    def file_pre_processing(self):
        HH_pro = self.HH.rename(columns={'herb1': 'node_from', 'herb2': 'node_to'})
        self.II['Combination'] = self.II[['ingre1', 'ingre2']].apply(lambda x: ''.join(sorted(x)), axis=1)
        II_pro = self.II.drop_duplicates(subset=['Combination', 'distance']).drop(columns=['Combination'])
        II_pro.columns = ['node_from', 'node_to', 'distance']
        DH_pro = self.DH.iloc[:, [0, 2, 3]].rename(
            columns={'Disease': 'node_from', 'Herb name': 'node_to', 'distance': 'distance'}).drop_duplicates()
        DI_pro = self.DI.iloc[:, [0, 4, 5]].rename(
            columns={'Disease name': 'node_from', 'Ingredient name': 'node_to', 'distance': 'distance'}).drop_duplicates()
        sum_file = pd.concat([II_pro, DH_pro, DI_pro,HH_pro], axis=0)
        
        return sum_file

    # detect the community by louvain_method
    def _detect_communities(self,sum_file):
        sum_file['distance1'] = max(sum_file['distance']) - sum_file['distance']
        sum_file = sum_file.iloc[:, [0, 1, 3]]
        sum_file.columns = ['node1', 'node2', 'weight']
        unique_values = pd.unique(sum_file[['node1', 'node2']].values.ravel())
        node_self = pd.DataFrame({'node1': unique_values, 'node2': unique_values, 'weight': 1})
        combined_data = pd.concat([node_self, sum_file], axis=0).drop_duplicates()
        
        # write the connection in PPI into network  object for calculation
        G = nx.Graph()
        for i in range(len(combined_data)):
            node1 = combined_data.iloc[i]['node1']
            node2 = combined_data.iloc[i]['node2']
            weight = combined_data.iloc[i]['weight']
            if not G.has_edge(node1, node2):
                G.add_edge(node1, node2, weight=weight)
        adj_matrix = nx.to_scipy_sparse_matrix(G)
        adj_matrix_np = adj_matrix.toarray()
        
        # communication detection
        communities, frames = louvain_method(adj_matrix_np)
        
        self.G, self.adj_matrix_np, self.communities, self.frames = G, adj_matrix_np, communities, frames
        
        return G, adj_matrix_np, communities, frames
    
        # In the provided code, the `frames` variable is being
        # used as an output from the `louvain_method` function.
        # The `frames` variable is returned along with the
        # `communities` variable from the `_detect_communities`
        # method.
        
        

    # prepare the community for disease as dataframe
    def _get_disease_communities(self,G,communities,DI,DH,disease_name):
        node_list = list(G.nodes())
        node_df = pd.DataFrame(node_list, columns=['Value'])
        node_index = node_list.index(disease_name)
        print("Node {} is at index:".format(disease_name), node_index)
        for i, s in enumerate(communities):
            if node_index in s:
                print("The element {} is in the {} set.".format(node_index, i + 1))
                break
        else:
            print("The element {} is not in any set.".format(node_index))
        set = communities[i]
        DHI_result = node_df.loc[node_df.index.isin(set)]
        subset_data_I= DI[DI['Ingredient name'].isin(DHI_result['Value'])]
        subset_data_H= DH[DH['Herb name'].isin(DHI_result['Value'])]
        return  DHI_result,subset_data_I,subset_data_H

    # prepare the community for all clusters as dataframe
    def _get_all_communities(self, G,communities):
        node_list = list(G.nodes())
        node_detect_df = pd.DataFrame(node_list, columns=['Value'])
        node_detect_df['community_id'] = None
        for community_id, community_set in enumerate(communities):
            node_detect_df.loc[node_detect_df.index.isin(community_set), 'community_id'] = community_id
        return node_detect_df
    
    # ADMET can be performed if necessary after get_disease_communities
    def _filter_admet(self, subset_data_I,ADMET):
        ADMET1 = ADMET[['tcmsp_ingredient_name','tcmsp_ingredient_ob','tcmsp_ingredient_drug_likeness']]
        
        # keep only edge connection with Z score and p value 
        subset_data_I_ADMET = pd.merge(subset_data_I, ADMET1, left_on='Ingredient name', right_on='tcmsp_ingredient_name',
                            how='left').drop_duplicates()
        
        # filter ingredient with ob(oral bioavailbility) and drug likely ness
        subset_data_I_ADMET = subset_data_I_ADMET[(subset_data_I_ADMET['tcmsp_ingredient_ob'] > 30) \
            & ( subset_data_I_ADMET['tcmsp_ingredient_drug_likeness'] > 0.18)]

        return subset_data_I_ADMET
    
    # detect the best distance for cut for max Q for 
    def _get_optimal_distance_threshold(self):
        sum_file_all = self.sum_file_all
        distance_value = list(sum_file_all['distance'])
        
        min_val = min(distance_value )
        max_val = max(distance_value )

        # set the loop distance threshold 
        start = np.ceil(min_val * 10) / 10  # 1.33 -> 1.4
        end = np.floor(max_val * 10) / 10    # 8.97 -> 8.9

        # 生成序列
        distance_threshold_list = list(np.arange(start, end + 0.05, 0.1))

        default_max_Q = 0.3
        default_distance_thresh = None
        print('start to loop for best distance cut{}'.format(distance_threshold_list))
        for distance_threshold in tqdm.tqdm(distance_threshold_list):
            sum_file = sum_file_all[(sum_file_all['distance'] < distance_threshold)|\
                (sum_file_all['node_from'] == self.disease_name) | \
                (sum_file_all['node_to'] == self.disease_name)]
            G, adj_matrix_np, communities, frames = self._detect_communities(sum_file)
            Q_value = list(frames[-1].items())[1][1]
            
            if Q_value >= default_max_Q:
                default_max_Q = Q_value
                print('Q value is {}'.format(default_max_Q))
                default_distance_thresh = distance_threshold
        
        if default_distance_thresh != None:
            print('Loop the distance from {} to {} with max Q = {} at distance cut of {}'.\
                format(start,end, default_max_Q, default_distance_thresh))
            return default_max_Q, default_distance_thresh
        else:
            print('The Q of all distance cut is less than 0.3 which is not ideal community')
            return None, max_val
            
            
    
    # 2. Perform the herb_synerger_finder 
    def get_herb_communities(self, ADMET_filter = False):
        max_Q, distance_threshold = self._get_optimal_distance_threshold()
        sum_file_all = self.sum_file_all
        
        sum_file = sum_file_all[(sum_file_all['distance'] < distance_threshold)|\
                (sum_file_all['node_from'] == self.disease_name) | \
                (sum_file_all['node_to'] == self.disease_name)]
        
        self.sum_file = sum_file
        
        G, adj_matrix_np, communities, frames = self._detect_communities(sum_file)
        DHI_result,subset_data_I,subset_data_H = self._get_disease_communities(G,communities,self.DI,self.DH,self.disease_name)
        self.subset_data_H_all,self.subset_data_I_all =  subset_data_H, subset_data_I
        
        # keep only edge connection with Z score and p value 
        subset_data_H_select = subset_data_H[(subset_data_H['Z-score'] < 0) & (subset_data_H['p-value'] < 0.05) ]
        
        node_detect_df = self._get_all_communities(G,communities)
        
        if ADMET_filter:
           ADMETf = 'source_data/ingredient_ADMET_Properties.xlsx'
           ADMET = pd.read_excel(ADMETf).drop_duplicates()
           subset_data_I_ADMET = self._filter_admet(subset_data_I,ADMET) 
           self.subset_data_H,self.subset_data_I_ADMET = subset_data_H_select,subset_data_I_ADMET
           subset_data_I = subset_data_I_ADMET
           subset_data_H = subset_data_H_select
        else:
            self.subset_data_H,self.subset_data_I =  subset_data_H_select, subset_data_I
        
        return G, communities, DHI_result, subset_data_I, subset_data_H_select, node_detect_df
    
    
        

    
class Detection_key_Module_multiple:
    def __init__(self, comb_data_ingre_list,comb_name_ingre_list,comb_data_herb_list, comb_name_herb_list, comb_data_merged_list, comb_name_merged_list):
        self.comb_data_ingre_list = comb_data_ingre_list
        self.comb_name_ingre_list = comb_name_ingre_list
        self.comb_data_herb_list = comb_data_herb_list
        self.comb_name_herb_list = comb_name_herb_list
        self.comb_data_merged_list = comb_data_merged_list
        self.comb_name_merged_list = comb_name_merged_list
        
        
    # merge files from multiple combination formula of herbs
    def _merge_cluster_files(self,comb_data_list,comb_name_list):
        new_comb_data_list = []
        for i,comb_one in enumerate(comb_data_list):
            comb_one['FJ'] = comb_name_list[i]
            new_comb_data_list.append(comb_one)
        sum_file = pd.concat(new_comb_data_list, axis=0)
        return sum_file
    
    # Data Processing Before Sankey Diagram
    def prepare_sankey_data(self):
        DIC_all_pd = self._merge_cluster_files(self.comb_data_ingre_list, self.comb_name_ingre_list)
        sankey_diagram_pd= DIC_all_pd.iloc[:, [0,2,4,5,13]].drop_duplicates()
        sankey_diagram_pd.columns = ['disease', 'herb', 'ingredient','distance','Prescription']
        return sankey_diagram_pd

    # Data Processing for_hierarchical_cluster
    def _prepare_cluster_matrix(self):
        # generate sankey_data
        sankey_data = self.sankey_data
        DI_pd = sankey_data.iloc[:, [0,2,3,4]].drop_duplicates()
        DI_pd.columns = ['node_from', 'node_to', 'distance', 'FJ']
        
        # merge all herb level distance
        DH_pd = self._merge_cluster_files(self.comb_data_herb_list, self.comb_name_herb_list)
        
        # prepare herb-level data
        DH_pd = DH_pd.iloc[:, [0,2,3,8]].drop_duplicates()
        DH_pd.columns = ['node_from', 'node_to', 'distance', 'FJ']
        
        # merge as whole matrix
        matrix_pre_pd = pd.concat([DI_pd,DH_pd])
        return matrix_pre_pd

    # prepare data for hierarchical cluster plot
    def prepare_hier_cluster_data(self):
        matrix_pre_pd = self._prepare_cluster_matrix()
        
        merged_pd = self._merge_cluster_files(self.comb_data_merged_list, self.comb_name_merged_list)
        
        merged_pd['adjusted_distance'] = max(merged_pd['distance'])-merged_pd['distance']
        
        merged_pd_combined = pd.concat([
            merged_pd.iloc[:, [0, 1, 4]],
            merged_pd.iloc[:, [1, 0, 4]].rename(columns={'node_to': 'node_from', 'node_from': 'node_to'})
        ]).drop_duplicates()
        
        merged_pd_combined.columns = ['node_from', 'node_to','distance']
        
        node_edge = merged_pd_combined[merged_pd_combined['node_from'].isin(matrix_pre_pd['node_to']) \
            & merged_pd_combined['node_to'].isin(matrix_pre_pd['node_to'])]
        
        unique_nodes = set(node_edge['node_from'].unique()).union(set(node_edge['node_to'].unique()))
        all_nodes = pd.DataFrame({
            'node_from': list(unique_nodes),
            'node_to': list(unique_nodes),
            'distance': max(merged_pd['distance'])
        })
        node_matrix_file = pd.concat([all_nodes, node_edge], axis=0).drop_duplicates().reset_index(drop=True)
        node_matrix_file = node_matrix_file.drop_duplicates(["node_from", "node_to"])
        scaler = MinMaxScaler()
        node_matrix_file['distance'] = scaler.fit_transform(node_matrix_file['distance'].values.reshape(-1, 1))
        final = node_matrix_file.pivot(index="node_from", columns="node_to", values="distance")
        for label in final.index:
            final.loc[label, label] = 1
        min_value = final.min().min()  # find minimum value
        matrix_filled = final.fillna(min_value)
        type_file = matrix_pre_pd.iloc[:, [1, 3]].drop_duplicates()
        
        return matrix_filled,min_value,type_file
    
    # prepare the file for figure plot and downstream analysis
    def prepare_downstream_data(self, path_to_save):
        self.sankey_data = self.prepare_sankey_data()
        
        self.sankey_data.to_csv("{}/sankey_diagram_file.csv".format(path_to_save), index=None)
        
        matrix_herb_type_file = self.sankey_data.iloc[:,[1,2]]
        matrix_filled,min_value,type_file = self.prepare_hier_cluster_data()
        
        self.matrix_filled = matrix_filled
        self.type_file = type_file
        
        matrix_filled.to_csv('{}/matrix_cluster_result.csv'.format(path_to_save))
        type_file.to_csv('{}/matrix_type_result.csv'.format(path_to_save), index=None)
        matrix_herb_type_file.to_csv('{}/matrix_herb_result.csv'.format(path_to_save), index=None)
        
    
