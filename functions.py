import pandas as pd

def clean_test(df):
    df.Age = df.Age.fillna(df.Age.mean())
    df.Fare = df.Fare.fillna(df.Fare.mean())
    embarked_dummies = pd.get_dummies(df['Embarked'], dtype=int)
    df = pd.concat([df.drop(columns=['Embarked']), embarked_dummies], axis=1)
    sex_dummies = pd.get_dummies(df['Sex'], drop_first=True, dtype=int)
    df = pd.concat([df.drop(columns=['Sex']), sex_dummies], axis=1)
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    return df