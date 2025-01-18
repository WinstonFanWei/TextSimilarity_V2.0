from tslearn.metrics import dtw_path_from_metric, soft_dtw
import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import math

from concurrent.futures import ProcessPoolExecutor

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class CompareFiles:
    def __init__(self, files_representations, paras, td, count):
        self.paras = paras
        self.td = td
        self.count = count
        self.files_representations = files_representations
        
    def process_row(self, row):
        file1 = row["file1"] + ".txt"
        file2 = row["file2"] + ".txt"
        # 调用对比方法并返回结果
        result = self.compare_two_files(file1, file2)
        return result

    def get_all_comparisons(self):
        output_path = os.path.join(self.paras["results_path"], "df_compare.csv")
        if os.path.exists(output_path):
            df_compare = pd.read_csv(output_path)
            print("文件已存在，已读取:", output_path)
        else:
            df_train = pd.read_csv(
                os.path.join(self.paras["file_path"], "train/similarity_scores.csv"),
                names=["file1", "file2", "GroundTruth"]
            )
            df_test = pd.read_csv(
                os.path.join(self.paras["file_path"], "test/similarity_scores.csv"),
                names=["file1", "file2", "GroundTruth"]
            )
            df_compare = pd.concat([df_train, df_test], ignore_index=True)  
            
        """使用多核并行处理 DataFrame"""
        with ProcessPoolExecutor(max_workers=self.paras["num_cores"]) as executor:
            # 提交任务到进程池并带进度条
            results = list(
                tqdm(
                    executor.map(self.process_row, [row for _, row in df_compare.iterrows()]),
                    desc=f'方案{self.count}计算中[{self.paras["model_config"][self.count]["token_class"]}]',
                    total=len(df_compare),
                )
            )
        
        df_compare[f'方案{self.count}'] = results
            
        # for index, row in tqdm(df_compare.iterrows(), desc=f'方案{self.count}计算中[{self.paras["model_config"][self.count]["token_class"]}]', total=len(df_compare)):
        #     df_compare.loc[index, f'方案{self.count}'] = self.compare_two_files(df_compare.loc[index, "file1"] + ".txt", df_compare.loc[index, "file2"] + ".txt").item()
            
        df_compare[f'方案{self.count}'] = round(df_compare[f'方案{self.count}'], 2)
        
        df_compare[f'method_{self.count}'] = round(df_compare[f'方案{self.count}'], 2)
        
        df_compare.to_csv(output_path, index=False, encoding='utf-8')

        df_compare[f'method_{self.count}'] = self.distance2similarity_and_normlayer(df_compare[f'method_{self.count}'])
        
        df_compare[f'method_{self.count}'] = round(df_compare[f'method_{self.count}'], 2)
        # df_compare = df_compare.drop(columns=[f'方案{self.count}'])

        df_compare.to_csv(output_path, index=False, encoding='utf-8')
        
    
    def compare_two_files(self, file1, file2):
        file1_represent = self.files_representations[file1]
        file2_represent = self.files_representations[file2]
        if(self.paras["model_config"][self.count]["token_representation_method"] == "TFIDF"):
            # 计算TF-IDF
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([file1_represent, file2_represent])
            
            # 计算余弦相似度
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return round(cosine_sim[0][0], 2)
        
        elif(self.paras["model_config"][self.count]["series_distance_method"] == 'DTW'):
            return self.compare_two_files_use_DTW(file1_represent, file2_represent)
        elif(self.paras["model_config"][self.count]["series_distance_method"] == 'DTW_changed'):
            pass
        elif(self.paras["model_config"][self.count]["token_class"] == 'document' and self.paras["model_config"][self.count]["series_distance_method"] == 'cosine'):
            return cosine_similarity([file1_represent], [file2_represent])[0][0]

    def compare_two_files_use_DTW(self, file1_represent, file2_represent):
        if self.paras["model_config"][self.count]["series_distance_method"] == "DTW":
            path, dtw_distance = dtw_path_from_metric(file1_represent, file2_represent, metric=self.td.get_token_distance)
        elif self.paras["model_config"][self.count]["series_distance_method"] == "Soft_DTW":
            file1_represent = np.array(file1_represent)
            file2_represent = np.array(file2_represent)
            dtw_distance = soft_dtw(file1_represent, file2_represent, gamma=1)
        elif self.paras["model_config"][self.count]["series_distance_method"] == "WDTW":
            pass
        DTW_standard = dtw_distance / len(path)
        # print(path)
        return DTW_standard

    def distance2similarity_and_normlayer(self, x):
        if self.paras["model_config"][self.count]["distance2similarity_method"] == "MinMaxScaler":
            x = x.values.reshape(-1, 1)
            x = 1 - x
            Q1 = np.percentile(x, 5)
            Q3 = np.percentile(x, 95)
            x[x < Q1] = Q1
            x[x > Q3] = Q3
            return MinMaxScaler().fit_transform(x).flatten()
        elif self.paras["model_config"][self.count]["distance2similarity_method"] == "Nonlinear":
            x = x.values.reshape(-1, 1)
            x = 1 - x
            Q1 = np.percentile(x, 5)
            Q3 = np.percentile(x, 95)
            x[x < Q1] = Q1
            x[x > Q3] = Q3
            x_scaled = MinMaxScaler().fit_transform(x).flatten()
            return - 0.5 * np.cos(np.pi * x_scaled) + 0.5
        elif self.paras["model_config"][self.count]["distance2similarity_method"] == "pass":
            return x
        