from gensim import corpora, models
import time

class LDA_model:
    def __init__(self):
        pass
    def LDA_model(self, data, paras):
        '''
        LDA模型训练
        '''
        # 构建Dictionary
        text_ls = []
        if paras["token_class"] == "word":
            for key, value in data.items():
                text_ls.append(value["file_content"])
            
            dictionary = corpora.Dictionary(text_ls)
            print("字典词数: ", len(dictionary))

        elif paras["token_class"] == "sentence":
            for key, value in data.items():
                for ls in value["file_sentences"]:
                    text_ls.append(ls)
                    
            dictionary = corpora.Dictionary(text_ls)
            print("字典词数: ", len(dictionary))

        elif paras["token_class"] == "paragraph":
            text_ls = []
            for key, value in data.items():
                for ls in value["file_paragraphs"]:
                    text_ls.append(ls)
            dictionary = corpora.Dictionary(text_ls)
            print("字典词数: ", len(dictionary))
        
        # 转换文档为词袋模型
        corpus = [dictionary.doc2bow(text) for text in text_ls]
        
        random_seed = 42

        # 构建 LDA 模型
        print("[LDA模型训练中]")
        start_time = time.time()
        lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=20, passes=40, random_state=random_seed)
        end_time = time.time()
        print("[LDA模型训练结束, 用时: " + str(round(end_time - start_time, 2)) + "s]")

        # 打印每个主题及其主题词
        print("\n[主题词展示]")
        for idx, topic in lda_model.print_topics(num_topics=20, num_words=10):
            print(f"主题 {idx}: {topic}")
     