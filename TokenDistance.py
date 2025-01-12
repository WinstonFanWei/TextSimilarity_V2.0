import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from datetime import datetime

class TokenDistance:
    def __init__(self, paras, count):
        self.paras = paras
        self.count = count
    
    def get_token_distance(self, token1, token2):
        if(self.paras["model_config"][self.count]["token_distance_method"] == "cosine"):
            return 1 - cosine_similarity([token1], [token2])[0][0] # cos越大距离越小
        if(self.paras["model_config"][self.count]["token_distance_method"] == "WMD"):
            return None
            # topic_distance_matrix
        else:
            return None