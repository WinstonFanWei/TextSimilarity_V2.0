import os
import Dataloader as Dl
import models.TokenRepresentation as TR

def main(data, paras):
    # 合并训练集和测试集数据
    all_data = data["train"] | data["test"]
    # 表示文本为向量
    representors = TR.represent(data, paras)



if __name__ == '__main__':
    print("-----------------------------------------------------------------------------------------------------------------")
    """ Main function. """
    
    # Parameters
    paras = {
        # 参数
        "file_path": "..\\data\\",

        # 开关
        "Debug_mode": True,

        # 模型
        "model_config": [
            {
                "token_representation_method": "LDA",
                "num_topics": 20,
                "passes": 40,
                "token_class": ["word", "sentence", "paragraph"],
                "distance_method": "cosine"
            }
        ]
    }

    """ prepare dataloader """
    data_path = {
        'train': os.path.join(paras["file_path"], "train\\docs"),
        'test': os.path.join(paras["file_path"], "test\\docs")
    }
    dataloader = Dl.Dataloader(data_path)
    data = dataloader.load()

    main(data, paras)