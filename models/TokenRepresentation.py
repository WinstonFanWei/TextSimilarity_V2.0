'''
功能：进行每个层级token的表示，输出向量。
'''
import models.LDA_model as lda_m
from sentence_transformers import SentenceTransformer

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
        
        elif self.paras["model_config"][self.count]["token_representation_method"] == 'sentence_bert':
            sentence_bert_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
            files_representations = {}
            for key, value in self.data.items():
                if self.paras["model_config"][self.count]["token_class"] == "sentence":
                    file_token_representation = [" ".join(words_list) for words_list in value["file_sentences"]]

                embeddings = sentence_bert_model.encode(file_token_representation)
                
                files_representations[key] = embeddings
                
            print(files_representations["test.txt"])
            return files_representations
        
        else:
            return None