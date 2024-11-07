import os
import Dataloader as Dl
import models.TokenRepresentation as TR

def main(data, paras):
    # 合并训练集和测试集数据
    all_data = data["train"] | data["test"]
    
    # 循环处理所有模型搭配
    for count in range(len(paras["model_config"])):
        representor = TR.representor(all_data, paras, count)
        files_representations = representor.get_token_representation()
        print(f"--------{count}次开始输出---------")
        print(files_representations["test.txt"])
        print(f"--------{count}次结束输出---------")


if __name__ == '__main__':
    print("----------------------------------------------------------------------------------------")
    """ Main function. """
    
    # Parameters
    paras = {
        # 参数
        "file_path": "/Users/winston/Desktop/Repository/TextSimilarity_V2.0/data",

        # 开关  
        "Debug_mode": True,

        # 模型
        "model_config": [
            {
                "token_representation_method": "LDA",
                "num_topics": 5,
                "passes": 10,
                "token_class": "word",
                "distance_method": "cosine"
            },
            {
                "token_representation_method": "LDA",
                "num_topics": 5,
                "passes": 10,
                "token_class": "sentence",
                "distance_method": "cosine"
            },
            {
                "token_representation_method": "LDA",
                "num_topics": 5,
                "passes": 10,
                "token_class": "paragraph",
                "distance_method": "cosine"
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