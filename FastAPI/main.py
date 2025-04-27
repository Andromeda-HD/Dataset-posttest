from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle


app = FastAPI()


with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)



try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    scaler = None

feature_names = [
    's80_s20_ratio', 'gini', 'headcount_ratio_40_median', 'headcount_ratio_50_median',
    'headcount_ratio_60_median', 'income_gap_ratio_60_median', 'income_inequality_level',
    'income_gap_ratio_50_median', 'decile9_share', 'income_gap_ratio_40_median',
    'poverty_gap_index_100', 'income_gap_ratio_2000', 'avg_shortfall_2000',
    'income_gap_ratio_3000', 'avg_shortfall_3000', 'income_gap_ratio_4000',
    'avg_shortfall_4000', 'headcount_ratio_100', 'poverty_gap_index_3000',
    'poverty_gap_index_2000', 'headcount_ratio_1000', 'poverty_gap_index_4000',
    'headcount_ratio_2000', 'poverty_gap_index_international_povline',
    'income_gap_ratio_1000', 'avg_shortfall_1000', 'poverty_gap_index_1000',
    'headcount_ratio_3000', 'headcount_ratio_upper_mid_income_povline',
    'poverty_gap_index_upper_mid_income_povline',
    'poverty_gap_index_lower_mid_income_povline', 'headcount_ratio_4000',
    'headcount_ratio_international_povline', 'headcount_ratio_lower_mid_income_povline',
    'income_gap_ratio_upper_mid_income_povline',
    'avg_shortfall_upper_mid_income_povline', 'decile9_thr', 'avg_shortfall_60_median',
    'avg_shortfall_50_median', 'avg_shortfall_40_median', 'decile9_avg', 'decile8_thr',
    'mean', 'decile8_avg', 'decile7_thr', 'decile7_avg', 'decile6_thr', 'decile6_avg',
    'median', 'decile5_avg', 'decile4_thr', 'decile4_avg', 'decile3_thr', 'decile3_avg',
    'decile2_thr', 'decile2_avg', 'decile1_thr', 'decile1_avg', 'decile8_share',
    'decile1_share', 'decile7_share', 'decile2_share', 'decile3_share', 'decile6_share',
    'decile4_share', 'decile5_share'
]


class InputData(BaseModel):
    s80_s20_ratio: float
    gini: float
    headcount_ratio_40_median: float
    headcount_ratio_50_median: float
    headcount_ratio_60_median: float
    income_gap_ratio_60_median: float
    income_inequality_level: float
    income_gap_ratio_50_median: float
    decile9_share: float
    income_gap_ratio_40_median: float
    poverty_gap_index_100: float
    income_gap_ratio_2000: float
    avg_shortfall_2000: float
    income_gap_ratio_3000: float
    avg_shortfall_3000: float
    income_gap_ratio_4000: float
    avg_shortfall_4000: float
    headcount_ratio_100: float
    poverty_gap_index_3000: float
    poverty_gap_index_2000: float
    headcount_ratio_1000: float
    poverty_gap_index_4000: float
    headcount_ratio_2000: float
    poverty_gap_index_international_povline: float
    income_gap_ratio_1000: float
    avg_shortfall_1000: float
    poverty_gap_index_1000: float
    headcount_ratio_3000: float
    headcount_ratio_upper_mid_income_povline: float
    poverty_gap_index_upper_mid_income_povline: float
    poverty_gap_index_lower_mid_income_povline: float
    headcount_ratio_4000: float
    headcount_ratio_international_povline: float
    headcount_ratio_lower_mid_income_povline: float
    income_gap_ratio_upper_mid_income_povline: float
    avg_shortfall_upper_mid_income_povline: float
    decile9_thr: float
    avg_shortfall_60_median: float
    avg_shortfall_50_median: float
    avg_shortfall_40_median: float
    decile9_avg: float
    decile8_thr: float
    mean: float
    decile8_avg: float
    decile7_thr: float
    decile7_avg: float
    decile6_thr: float
    decile6_avg: float
    median: float
    decile5_avg: float
    decile4_thr: float
    decile4_avg: float
    decile3_thr: float
    decile3_avg: float
    decile2_thr: float
    decile2_avg: float
    decile1_thr: float
    decile1_avg: float
    decile8_share: float
    decile1_share: float
    decile7_share: float
    decile2_share: float
    decile3_share: float
    decile6_share: float
    decile4_share: float
    decile5_share: float

# Endpoint root
@app.get("/")
def home():
    return {"message": "API is running!"}

# Endpoint predict
@app.post("/predict")
def predict(data: InputData):
    input_array = np.array([[getattr(data, feature) for feature in feature_names]])
    
    if scaler:
        input_array = scaler.transform(input_array)
    
    prediction = model.predict(input_array)
    result = prediction.tolist()

    return {"prediction": result}

