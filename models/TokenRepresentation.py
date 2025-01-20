'''
功能：进行每个层级token的表示，输出向量。
'''
import models.LDA_model as lda_m
import models.Word2Vec_model as w2v_m
from sentence_transformers import SentenceTransformer
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class representor:
    def __init__(self, data, paras, count):
        self.paras = paras
        self.data = data
        self.count = count

    def get_token_representation(self):
        if self.paras["model_config"][self.count]["token_representation_method"] == 'LDA':
            lda_model = lda_m.LDA_model()
            lda_model.train_LDA_model(self.data, self.paras, self.count)
            return lda_model.token_represent(self.data, self.paras, self.count)
        
        elif self.paras["model_config"][self.count]["token_representation_method"] == 'Word2Vec':
            word2vec_model = w2v_m.Word2Vec_model()
            word2vec_model.train_Word2Vec_model(self.data, self.paras, self.count)
            return word2vec_model.token_represent(self.data, self.paras, self.count)
        
        elif self.paras["model_config"][self.count]["token_representation_method"] == 'BERT':
            sentence_bert_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
            files_representations = {}
            for key, value in self.data.items():
                if self.paras["model_config"][self.count]["token_class"] == "word":
                    file_token_representation = value["file_content"]
                    embeddings = sentence_bert_model.encode(file_token_representation)

                elif self.paras["model_config"][self.count]["token_class"] == "sentence":
                    file_token_representation = [" ".join(words_list) for words_list in value["file_sentences"]]
                    embeddings = sentence_bert_model.encode(file_token_representation)

                elif self.paras["model_config"][self.count]["token_class"] == "paragraph":
                    file_token_representation = [" ".join(words_list) for words_list in value["file_paragraphs"]]
                    embeddings = sentence_bert_model.encode(file_token_representation)

                elif self.paras["model_config"][self.count]["token_class"] == "document":
                    file_token_representation = [" ".join(words_list) for words_list in value["file_sentences"]]
                    embeddings = sentence_bert_model.encode(file_token_representation)
                    # 转为NumPy数组
                    embeddings_np = np.array(embeddings)
                    # 计算每列的平均值
                    embeddings = np.mean(embeddings_np, axis=0)
                
                files_representations[key] = embeddings
                
            # print(files_representations["test.txt"])
            return files_representations
        
        elif self.paras["model_config"][self.count]["token_representation_method"] == 'TFIDF':
            files_representations = {}
            for key, value in self.data.items():
                file_token_representation = " ".join(value["file_content"])
                files_representations[key] = file_token_representation

            return files_representations

        else:
            return None