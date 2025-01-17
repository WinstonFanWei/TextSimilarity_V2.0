import os
import logging
from sklearn.metrics import f1_score

def compute_metrics(df_compare, paras, len):
    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler('log/debug.log'),
                        logging.StreamHandler()
                    ])
    logger = logging.getLogger(__name__)

    for i in range(len):
        r_model = paras["model_config"][i]["token_representation_method"]
        token_class = paras["model_config"][i]["token_class"]
        sd = paras["model_config"][i]["series_distance_method"]
        d2s = paras["model_config"][i]["distance2similarity_method"]

        rmse = str(compute_rmse(df_compare[f"method_{i}"], df_compare["GroundTruth"]))
        corr = str(compute_correlation(df_compare[f"method_{i}"], df_compare["GroundTruth"]))
        f1 = str(compute_f1(df_compare[f"method_{i}"], df_compare["GroundTruth"]))

        message = f"\n[method{i} ~ R-model:{r_model:<12} token:{token_class:<10} SD:{sd:<8} D2S:{d2s:<15}]" + f"\nRMSE: {rmse}" + f"\nCORR: {corr}" + f"\nF1-score: {f1} \n"

        logger.debug(message)


def rmse(predictions, targets):
    return (((predictions - targets) ** 2).mean()) ** 0.5

def compute_rmse(a, b):
    return round(rmse(a, b), 4)
    
def compute_correlation(a, b):
    return round(a.corr(b), 4)

def compute_f1(a, b):
    return round(f1_score((a > 0.5).astype(int), (b > 0.5).astype(int)), 4)
    