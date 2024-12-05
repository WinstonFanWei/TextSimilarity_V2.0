from gensim import corpora, models
import time

class LDA_model:
    def __init__(self):
        self.lda_model = None
        self.dictionary = None
    def train_LDA_model(self, data, paras, count):
        '''
        LDA模型训练
        '''
        # 构建Dictionary
        text_ls = []
        
        print(data["test.txt"])
        
        '''
        {'file_path': 'C:/Users/Winston/Desktop/Repository/TextSimilarity_V2.0/data\\train/docs\\test.txt', 'file_content': ['accept', 'mine', 'opinion', 'oh', 'girl', 'sing', 'wonder'], 'file_sentences': [['accept', 'mine'], ['opinion'], ['oh', 'girl', 'sing'], ['wonder']], 'file_paragraphs': [['accept', 'mine', 'opinion', 'oh', 'girl', 'sing', 'wonder']]}
        '''
        
        if paras["model_config"][count]["token_class"] == "word" or paras["model_config"][count]["token_class"] == "document":
            for key, value in data.items():
                text_ls.append(value["file_content"])
            
            self.dictionary = corpora.Dictionary(text_ls)
            print("字典词数: ", len(self.dictionary))

        elif paras["model_config"][count]["token_class"] == "sentence":
            for key, value in data.items():
                for ls in value["file_sentences"]:
                    text_ls.append(ls)
                    
            self.dictionary = corpora.Dictionary(text_ls)
            print("字典词数: ", len(self.dictionary))

        elif paras["model_config"][count]["token_class"] == "paragraph":
            for key, value in data.items():
                for ls in value["file_paragraphs"]:
                    text_ls.append(ls)
            self.dictionary = corpora.Dictionary(text_ls)
            print("字典词数: ", len(self.dictionary))
        
        # 转换文档为词袋模型
        corpus = [self.dictionary.doc2bow(text) for text in text_ls]
        
        random_seed = 42

        # 构建 LDA 模型
        print("[LDA模型训练中]")
        start_time = time.time()
        self.lda_model = models.LdaModel(corpus, id2word=self.dictionary, num_topics=paras["model_config"][count]["num_topics"], passes=paras["model_config"][count]["passes"], random_state=random_seed)
        end_time = time.time()
        print("[LDA模型训练结束, 用时: " + str(round(end_time - start_time, 2)) + "s]")

        # 打印每个主题及其主题词
        # print("\n[主题词展示]")
        # self.lda_model.print_topics(num_topics=20, num_words=10)

    def token_represent(self, data, paras, count):
        # 得到文档每个token的topic概率
        files_representations = {}
        for key, value in data.items():
            file_token_representation = []

            if paras["model_config"][count]["token_class"] == "word":
                for word in value["file_content"]:
                    token_topic = self.lda_model.get_term_topics(self.dictionary.token2id.get(word), minimum_probability=0)
                    token_topic = self.normalize_token_represent(token_topic)
                    file_token_representation.append(token_topic)

            elif paras["model_config"][count]["token_class"] == "sentence":
                for sentence in value["file_sentences"]:
                    token_topic = self.lda_model.get_document_topics(self.dictionary.doc2bow(sentence), minimum_probability=0, minimum_phi_value=0)
                    token_topic = self.normalize_token_represent(token_topic)
                    file_token_representation.append(token_topic)

            elif paras["model_config"][count]["token_class"] == "paragraph":
                for paragraph in value["file_paragraphs"]:
                    token_topic = self.lda_model.get_document_topics(self.dictionary.doc2bow(paragraph), minimum_probability=0, minimum_phi_value=0)
                    token_topic = self.normalize_token_represent(token_topic)
                    file_token_representation.append(token_topic)
                    
            elif paras["model_config"][count]["token_class"] == "document":
                file_token_representation = self.lda_model.get_document_topics(self.dictionary.doc2bow(value["file_content"]), minimum_probability=0, minimum_phi_value=0)
                file_token_representation = self.normalize_token_represent(file_token_representation)

            files_representations[key] = file_token_representation
            
        print(files_representations["test.txt"])
        '''   
        [[0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.342, 0.008, 0.008, 0.008, 0.008, 0.342, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008], [0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.512, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013], [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]]
        '''

        return files_representations
    
    def normalize_token_represent(self, token_topic):
        token_topic = [prob for _, prob in token_topic]
        total = sum(token_topic)
        normalized_token_topic = [round(prob / total, 3) if total > 0 else 0 for prob in token_topic]
        return normalized_token_topic
     