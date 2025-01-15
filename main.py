import os
import Dataloader as Dl
import models.TokenRepresentation as TR
from TokenDistance import TokenDistance
from CompareFiles import CompareFiles
import Utils
import pandas as pd
import time
import logging

from datetime import datetime

import locale
import os

locale.setlocale(locale.LC_ALL, '')
os.environ['LC_ALL'] = 'en_US.UTF-8'
os.environ['LANG'] = 'en_US.UTF-8'

def main(data, paras):
    # 合并训练集和测试集数据
    all_data = data["train"] | data["test"]
    
    # 数据格式
    # print(all_data["test.txt"])
    
    # sentences = [" ".join(words) for words in all_data["test.txt"]["file_sentences"]]
    # print(sentences)
    
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
    start_time = time.time()

    # 打印开始时间
    print(f"程序开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    """ Main function. """
    
    # Parameters
    paras = {
        # 参数
        # "file_path": "C:/Users/Winston/Desktop/Repository/TextSimilarity_V2.0/data", 
        # "results_path": "C:/Users/Winston/Desktop/Repository/TextSimilarity_V2.0/results", 
        "file_path": "/Users/winston/Desktop/Repository/TextSimilarity_V2.0/data",
        "results_path": "/Users/winston/Desktop/Repository/TextSimilarity_V2.0/results",

        # 开关
        "Debug_mode": True,
        
        # CPU核心工作数 windows最大20 mac最大10
        "num_cores": 5,

        # 模型
        "model_config": [
            {
                "token_representation_method": "Word2Vec",
                "token_class": "document",
                "token_distance_method": "cosine",
                "series_distance_method": "DTW",
                "distance2similarity_method": "MinMaxScaler"
            },
            {
                "token_representation_method": "Word2Vec",
                "token_class": "sentence",
                "token_distance_method": "cosine",
                "series_distance_method": "DTW",
                "distance2similarity_method": "MinMaxScaler"
            },
            {
                "token_representation_method": "Word2Vec",
                "token_class": "paragraph",
                "token_distance_method": "cosine",
                "series_distance_method": "DTW",
                "distance2similarity_method": "MinMaxScaler"
            },
            # {
            #     "token_representation_method": "TFIDF",
            #     "token_class": "TFIDF_document",
            #     "series_distance_method": "cosine",
            #     "distance2similarity_method": "pass",
            # },
            # {
            #     "token_representation_method": "LDA",
            #     "num_topics": 50,
            #     "passes": 100,
            #     "token_class": "document",
            #     "series_distance_method": "cosine",
            #     "distance2similarity_method": "pass"
            # },
            # {
            #     "token_representation_method": "LDA",
            #     "num_topics": 20,
            #     "passes": 40,
            #     "token_class": "word",
            #     "token_distance_method": "cosine",
            #     "series_distance_method": "DTW",
            #     "distance2similarity_method": "MinMaxScaler"
            # },
            # {
            #     "token_representation_method": "LDA",
            #     "num_topics": 50,
            #     "passes": 100,
            #     "token_class": "sentence",
            #     "token_distance_method": "cosine",
            #     "series_distance_method": "DTW",
            #     "distance2similarity_method": "MinMaxScaler",
            # },
            # {
            #     "token_representation_method": "LDA",
            #     "num_topics": 50,
            #     "passes": 100,
            #     "token_class": "paragraph",
            #     "token_distance_method": "cosine",
            #     "series_distance_method": "DTW",
            #     "distance2similarity_method": "MinMaxScaler",
            # },
            # {
            #     "token_representation_method": "sentence_bert",
            #     "token_class": "sentence",
            #     "token_distance_method": "cosine",
            #     "series_distance_method": "DTW",
            #     "distance2similarity_method": "MinMaxScaler",
            # }
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
    
    end_time = time.time()

    execution_time = end_time - start_time
    print(f"程序运行时间：{execution_time:.2f} 秒")
    
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('log/debug.log'),
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger(__name__)

    message = f"\nRUN TIME: {round(execution_time / 60, 2)} minutes\n----------------------------------------------------------------------------------------" 
    
    logger.debug(message)