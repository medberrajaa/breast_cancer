from sklearn.datasets import load_breast_cancer
import pandas as pd

breast_cancer_data = load_breast_cancer()

df = pd.DataFrame(breast_cancer_data.data, columns=breast_cancer_data.feature_names)
df.to_csv("./data/breat_cancer.txt",index=False)