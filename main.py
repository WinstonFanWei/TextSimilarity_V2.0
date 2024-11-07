import os
import Dataloader as Dl
import models.TokenRepresentation as TR
from TokenDistance import TokenDistance
from CompareFiles import CompareFiles
import Utils
import pandas as pd

def main(data, paras):
    # 合并训练集和测试集数据
    all_data = data["train"] | data["test"]
    
    # 循环处理所有模型搭配
    for count in range(len(paras["model_config"])):
        representor = TR.representor(all_data, paras, count)
        files_representations = representor.get_token_representation()
        td = TokenDistance(paras, count)
        compareFiles = CompareFiles(files_representations, paras, td, count)
        compareFiles.get_all_comparisons()

    # 指标评估
    output_path = os.path.join(paras["results_path"], "df_compare.csv")
    df_compare = pd.read_csv(output_path)
    Utils.compute_metrics(df_compare, paras, len(paras["model_config"]))


if __name__ == '__main__':
    print("----------------------------------------------------------------------------------------")
    """ Main function. """
    
    # Parameters
    paras = {
        # 参数
        "file_path": "/Users/winston/Desktop/Repository/TextSimilarity_V2.0/data",
        "results_path": "/Users/winston/Desktop/Repository/TextSimilarity_V2.0/results",

        # 开关
        "Debug_mode": True,

        # 模型
        "model_config": [
            # {
            #     "token_representation_method": "LDA",
            #     "num_topics": 5,
            #     "passes": 10,
            #     "token_class": "word",
            #     "token_distance_method": "cosine",
            #     "series_distance_method": "DTW",
            #     "distance2similarity_method": "2*(1-x)"
            # },
            {
                "token_representation_method": "LDA",
                "num_topics": 5,
                "passes": 10,
                "token_class": "sentence",
                "token_distance_method": "cosine",
                "series_distance_method": "DTW",
                "distance2similarity_method": "2*(1-x)"
            },
            {
                "token_representation_method": "LDA",
                "num_topics": 5,
                "passes": 10,
                "token_class": "paragraph",
                "token_distance_method": "cosine",
                "series_distance_method": "DTW",
                "distance2similarity_method": "2*(1-x)"
            }
        ]
    }

    """ prepare dataloader """
    data_path = {
        'train': os.path.join(paras["file_path"], "train/docs"),
        'test': os.path.join(paras["file_path"], "test/docs")
    }
    dataloader = Dl.Dataloader(data_path)
    data = dataloader.load()

    main(data, paras)

    print("----------------------------------------------------------------------------------------")