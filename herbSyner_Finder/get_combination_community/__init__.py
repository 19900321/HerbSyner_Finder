from pathlib import Path

# from . import (
#     #calculate_synergy_scores,
#     community_detection,
#     #plot_hier_cluster_heatmap,
#     #plot_sankey_diagram
#     )

# from .community_detection import (
#     Herb_Comb_Community,
#     Detection_key_Module_multiple
# )

from pathlib import Path

# 导入子模块
from .community_detection import (
    Herb_Comb_Community,
    Detection_key_Module_multiple,
    # 添加其他需要的函数/类
)

# 如果有其他需要导入的模块
# from .calculate_synergy_scores import calculate_synergy
# from .plot_hier_cluster_heatmap import plot_heatmap

__all__ = [
    'Herb_Comb_Community',
    'Detection_key_Module_multiple',
    # 'calculate_synergy',
    # 'plot_heatmap',
]