import time
from gensim.models import Word2Vec
from sklearn.decomposition import NMF

class Word2Vec_model:
    def __init__(self):
        self.word2vec_model = None
    def train_Word2Vec_model(self, data, paras, count):
        '''
        Word2Vec模型训练
        '''
        text_ls = []
        for key, value in data.items():
            text_ls.append(value["file_content"])

        # 训练word2Vec模型
        print("[Word2Vec模型训练中]")
        random_seed = 42
        start_time = time.time()
        self.word2vec_model = Word2Vec(
            text_ls, 
            vector_size=paras["model_config"][count]["vector_size"], 
            window=paras["model_config"][count]["window"], 
            min_count=1, 
            workers=4, 
            seed=random_seed
        )
        end_time = time.time()
        print("[Word2Vec模型训练结束, 用时: " + str(round(end_time - start_time, 2)) + "s]")

    def token_represent(self, data, paras, count):
        # 得到文档每个token的向量表达
        files_representations = {}
        for key, value in data.items():
            file_token_representation = []

            if paras["model_config"][count]["token_class"] == "word":
                for word in value["file_content"]:
                    word_vector = self.word2vec_model.wv[word]
                    
                    file_token_representation.append(word_vector)

            elif paras["model_config"][count]["token_class"] == "sentence":
                for sentence in value["file_sentences"]:
                    sentence_vector = [0] * self.word2vec_model.vector_size
                    sentence_lenth = len(sentence)
                    for word in sentence:
                        if word in self.word2vec_model.wv:
                            sentence_vector = [a + b for a, b in zip(sentence_vector, self.word2vec_model.wv[word])]
                    sentence_vector = [a / sentence_lenth for a in sentence_vector]

                    file_token_representation.append(sentence_vector)

            elif paras["model_config"][count]["token_class"] == "paragraph":
                for paragraph in value["file_sentences"]:
                    paragraph_vector = [0] * self.word2vec_model.vector_size
                    paragraph_lenth = len(paragraph)
                    for word in paragraph:
                        if word in self.word2vec_model.wv:
                            paragraph_vector = [a + b for a, b in zip(paragraph_vector, self.word2vec_model.wv[word])]

                    paragraph_vector = [a / paragraph_lenth for a in paragraph_vector]

                    file_token_representation.append(paragraph_vector)
                    
            elif paras["model_config"][count]["token_class"] == "document":
                document_vector = [0] * self.word2vec_model.vector_size
                document_lenth = len(value["file_content"])
                for word in value["file_content"]:
                    document_vector = [a + b for a, b in zip(document_vector, self.word2vec_model.wv[word])]
                document_vector = [a / document_lenth for a in document_vector]
                    
                file_token_representation = document_vector

            files_representations[key] = file_token_representation
            
        print(files_representations["test.txt"])
        '''   
        [[0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.342, 0.008, 0.008, 0.008, 0.008, 0.342, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008], [0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.512, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013], [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025], [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025]]
        '''

        return files_representations
     