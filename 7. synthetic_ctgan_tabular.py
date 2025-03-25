import pandas as pd
from ctgan import CTGAN
from table_evaluator import load_data, TableEvaluator


def train_ctgan(data, categorical_features):
    ctgan = CTGAN(
        batch_size=130,  
        generator_dim=(256, 256), 
        discriminator_dim=(256, 256),  
        epochs=300,  
        verbose=True
    )
    ctgan.fit(data, categorical_features, epochs=300)  
    samples = ctgan.sample(2000)  
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
save_to_csv(synthetic_data,'synthetic.csv')
# synthetic = pd.read_csv('synthetic.csv')
# compare_data(df, synthetic, [])

