import os
import Dataloader as Dl
import models.TokenRepresentation as TR


def main(data, paras):
    # 合并训练集和测试集数据
    all_data = data["train"] | data["test"]
    
    # 循环处理所有模型搭配
    for count in range(len(paras["model_config"])):
        print(count)
        representor = TR.representor(all_data, paras, count)
        lda_model = representor.get_token_representation()
        print(lda_model)


if __name__ == '__main__':
    print("-----------------------------------------------------------------------------------------------------------------")
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
                "num_topics": 50,
                "passes": 100,
                "token_class": "word",
                "distance_method": "cosine"
            },
            {
                "token_representation_method": "LDA",
                "num_topics": 30,
                "passes": 3,
                "token_class": "sentence",
                "distance_method": "cosine"
            },
            {
                "token_representation_method": "LDA",
                "num_topics": 30,
                "passes": 5,
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

    print("结束")