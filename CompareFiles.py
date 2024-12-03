from tslearn.metrics import dtw_path_from_metric
import pandas as pd
import os
from tqdm import tqdm

class CompareFiles:
    def __init__(self, files_representations, paras, td, count):
        self.paras = paras
        self.td = td
        self.count = count
        self.files_representations = files_representations

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
            
        for index, row in tqdm(df_compare.iterrows(), desc=f'方案{self.count}计算中[{self.paras["model_config"][self.count]["token_class"]}]', total=len(df_compare)):
            df_compare.loc[index, f'方案{self.count}'] = self.compare_two_files(df_compare.loc[index, "file1"] + ".txt", df_compare.loc[index, "file2"] + ".txt").item()
            
        df_compare[f'method_{self.count}'] = round(df_compare[f'方案{self.count}'], 2)

        df_compare[f'method_{self.count}'] = round(self.distance2similarity(df_compare[f'method_{self.count}']), 2)
        df_compare = df_compare.drop(columns=[f'方案{self.count}'])

        df_compare.to_csv(output_path, index=False, encoding='utf-8')
        
    
    def compare_two_files(self, file1, file2):
        file1_represent = self.files_representations[file1]
        file2_represent = self.files_representations[file2]
        if(self.paras["model_config"][self.count]["series_distance_method"] == 'DTW'):
            return self.compare_two_files_use_DTW(file1_represent, file2_represent)
        elif(self.paras["model_config"][self.count]["series_distance_method"] == 'DTW_changed'):
            pass
    
    def compare_two_files_use_DTW(self, file1_represent, file2_represent):
        path, dtw_distance = dtw_path_from_metric(file1_represent, file2_represent, metric=self.td.get_token_distance)
        DTW_standard = dtw_distance / len(path)
        # print(path)
        return DTW_standard
    
    def distance2similarity(self, x):
        if self.paras["model_config"][self.count]["distance2similarity_method"] == "(1-2*x)":
            return (1 - 2 * x)
        else:
            pass
        