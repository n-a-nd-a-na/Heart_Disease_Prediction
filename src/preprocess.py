import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):

    le_sex = LabelEncoder()
    le_exercise = LabelEncoder()

    df['Sex'] = le_sex.fit_transform(df['Sex'])
    df['ExerciseAngina'] = le_exercise.fit_transform(df['ExerciseAngina'])



    new_df = pd.get_dummies(df, columns = ['ChestPainType','RestingECG','ST_Slope'], drop_first = True, dtype = int)


    X = new_df.drop(columns = 'HeartDisease')
    y = new_df['HeartDisease']

    return X, y