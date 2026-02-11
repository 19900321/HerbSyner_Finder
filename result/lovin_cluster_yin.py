import networkx as nx
import pandas as pd
from communities.algorithms import louvain_method
from sklearn.preprocessing import MinMaxScaler


def scale_data(II,DI):
    relation_sums = II.groupby('node_from')['distance'].sum().add(
        II.groupby('node_to')['distance'].sum(), fill_value=0
    )
    relation_sums = pd.DataFrame(relation_sums).reset_index()
    relation_sums.columns = ['Compound', 'TotalValue']
    total_values_dict = dict(zip(relation_sums['Compound'], relation_sums['TotalValue']))
    # 使用apply来计算每一行的total_value
    II['total_value1'] = II['node_from'].apply(lambda x: total_values_dict.get(x, 0))
    II['total_value2'] = II['node_to'].apply(lambda x: total_values_dict.get(x, 0))
    # 计算平均关系值
    II['AverageRelationValue'] = II.apply(
        lambda row: row['distance'] / (row['total_value1'] * row['total_value2']) if row['distance'] != 0 else 0,
        axis=1
    )
    DI['total_value1'] = DI['node_to'].apply(lambda x: total_values_dict.get(x, 0))
    # 计算平均关系值
    DI['AverageRelationValue'] = DI.apply(
        lambda row: row['distance'] / row['total_value1'] if row['distance'] != 0 else 0,
        axis=1
    )

    II['distance'] = (II['AverageRelationValue'] - II['AverageRelationValue'].min()) / (
                II['AverageRelationValue'].max() - II['AverageRelationValue'].min())
    DI['distance'] = (DI['AverageRelationValue'] - DI['AverageRelationValue'].min()) / (
            DI['AverageRelationValue'].max() - DI['AverageRelationValue'].min())

    II = II.iloc[:, [0, 1, 2]]
    DI = DI.iloc[:, [0, 1, 2]]
    return II,DI


def file_pre_processing(HIf, HHf, IIf, DHf, DIf):
    HI = pd.read_csv(HIf).drop_duplicates()
    HH = pd.read_csv(HHf).drop_duplicates()
    II = pd.read_csv(IIf).drop_duplicates()
    DH = pd.read_csv(DHf).drop_duplicates()
    DI = pd.read_csv(DIf).drop_duplicates()
    HI = HI.rename(columns={'herb': 'node_from', 'ingredient': 'node_to'})
    HI['distance'] = 0
    HH = HH.rename(columns={'herb1': 'node_from', 'herb2': 'node_to'})
    II['Combination'] = II[['ingre1', 'ingre2']].apply(lambda x: ''.join(sorted(x)), axis=1)
    II = II.drop_duplicates(subset=['Combination', 'distance']).drop(columns=['Combination'])
    II.columns = ['node_from', 'node_to', 'distance']
    DH = DH.iloc[:, [0, 2, 3]].rename(
        columns={'Disease': 'node_from', 'Herb name': 'node_to', 'distance': 'distance'}).drop_duplicates()
    DI = DI.iloc[:, [0, 4, 5]].rename(
        columns={'Disease name': 'node_from', 'Ingredient name': 'node_to', 'distance': 'distance'}).drop_duplicates()
    # II,DI= scale_data(II,DI)
    # HH,DH = scale_data(HH,DH)
    # sum_file = pd.concat([HI, II, DH, DI,HH], axis=0) # yin change
    sum_file = pd.concat([II, DH, DI,HH], axis=0)
    return sum_file,DH,DI

def detect_communities(data):
    data['distance1'] = max(data['distance']) - data['distance']# yin change to max(data['distance']), rather than 1
    data = data.iloc[:, [0, 1, 3]]
    data.columns = ['node1', 'node2', 'weight']
    unique_values = pd.unique(data[['node1', 'node2']].values.ravel())
    data1 = pd.DataFrame({'node1': unique_values, 'node2': unique_values, 'weight': 1})
    data = pd.concat([data1, data], axis=0).drop_duplicates()
    G = nx.Graph()
    for i in range(len(data)):
        node1 = data.iloc[i]['node1']
        node2 = data.iloc[i]['node2']
        weight = data.iloc[i]['weight']
        if not G.has_edge(node1, node2):
            G.add_edge(node1, node2, weight=weight)
    adj_matrix = nx.to_scipy_sparse_matrix(G)
    adj_matrix_np = adj_matrix.toarray()
    communities, frames = louvain_method(adj_matrix_np)
    return  G,  adj_matrix_np, communities, frames




def communities_result(G,communities,DI,DH):
    my_list = list(G.nodes())
    df = pd.DataFrame(my_list, columns=['Value'])
    node_index = my_list.index("Cough Variant Asthma")
    print("Node 'Cough Variant Asthma' is at index:", node_index)
    for i, s in enumerate(communities):
        if node_index in s:
            print("元素 {} 在第 {} 个集合中.".format(node_index, i + 1))
            break
    else:
        print("元素 {} 不在任何集合中.".format(node_index))
    set = communities[i]
    result = df.loc[df.index.isin(set)]
    subset_data1= DI[DI['node_to'].isin(result['Value'])]
    subset_data2= DH[DH['node_to'].isin(result['Value'])]
    return  result,subset_data1,subset_data2


def deal_admet_file(DIC,ADMET):
    ADMET1 = ADMET[['mol_inchikey_dict', 'tcmsp_ingredient_name', 'tcmsp_ingredient_ob', 'tcmsp_ingredient_bbb',
                    'tcmsp_ingredient_drug_likeness']]

    DIC_ADMET = pd.merge(DIC, ADMET1, left_on='node_to', right_on='tcmsp_ingredient_name',
                         how='left').drop_duplicates()
    DIC_ADMET = DIC_ADMET[(DIC_ADMET['tcmsp_ingredient_ob'] > 30) & (DIC_ADMET['tcmsp_ingredient_drug_likeness'] > 0.18)]

    return DIC_ADMET


HIf='herb_ingredient_pairs_pd.csv'
HHf='herb_herb_dis_pd.csv'
IIf='ingre_ingre_dis_pd.csv'
DHf='herb_disease_pd.csv'
DIf='herb_disease_ingre_pd.csv'
SGsum_file ,DH, DI = file_pre_processing(HIf, HHf, IIf, DHf, DIf)



##统计
value_counts = SGsum_file['distance'].value_counts()
SGstatistical_table = value_counts.reset_index()
SGstatistical_table.columns = ['Value', 'Count']



##裁剪
SGsum_file06 = SGsum_file[(SGsum_file['distance'] < 1.3)|(SGsum_file['node_from'] == 'Cough Variant Asthma') | (SGsum_file['node_to'] == 'Cough Variant Asthma')]
G, adj_matrix_np, communities, frames = detect_communities(SGsum_file06)
print(list(frames[-1].items())[1])



##聚类结果
result,subset_data_I,subset_data_H = communities_result(G,communities,DI,DH,)
# ADMETf='intergration/data/herb_ingre_info/herb_ingre_tcmsp_pd.xlsx'
# ADMET = pd.read_excel(ADMETf).drop_duplicates()
# DI_ADMET= deal_admet_file(subset_data_I,ADMET)

# check
[i for i in [ 'luteolin','berberine', 'kaempferol', 'quercetin'] if i in subset_data_I.iloc[:, 1].values]


[i for i in ['lauric acid', 'luteolin','berberine', 'kaempferol', 'quercetin'] if i in SGsum_file06.iloc[:, 1].values]

[i for i in ['luteolin','berberine', 'kaempferol', 'quercetin'] if (i  in SGsum_file06['node_from'].values) or (i in SGsum_file06['node_to'].values)]


SGsum_file06.loc[(SGsum_file06['node_from'] == 'luteolin') | (SGsum_file06['node_to'] == 'luteolin'),:]


SGsum_file06.loc[(SGsum_file06['node_from'] == 'berberine') | (SGsum_file06['node_to'] == 'berberine'),:]


SGsum_file06.loc[(SGsum_file06['node_from'] == 'kaempferol') | (SGsum_file06['node_to'] == 'kaempferol'),:]


SGsum_file06.loc[(SGsum_file06['node_from'] ==  'quercetin') | (SGsum_file06['node_to'] ==  'quercetin'),:]