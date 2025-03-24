import pandas as pd
from ctgan import CTGAN
from table_evaluator import load_data, TableEvaluator


def train_ctgan(data, categorical_features):
    ctgan = CTGAN(batch_size=250,generator_dim=(128, 128),discriminator_dim=(128, 128),verbose=True)
    ctgan.fit(data, categorical_features, epochs = 10)
    samples = ctgan.sample(1000)
    print(samples.head())
    return samples

def compare_data(real, synthetic, categorical_features):
    print(real.shape, synthetic.shape)
    table_evaluator =  TableEvaluator(real, synthetic, cat_cols= categorical_features)
    table_evaluator.visual_evaluation()

def save_to_csv(data,name):
    data.to_csv(name, index=False) 


df = pd.read_csv('COMBO2.csv')
df = df.drop('Filename', axis=1)
synthetic_data =  train_ctgan(df,[])
save_to_csv(df,'synthetic.csv')
# synthetic = pd.read_csv('synthetic.csv')
# compare_data(df, synthetic, [])

