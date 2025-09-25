import pandas as pd

def import_train_data():
    df = pd.read_csv('data/train1.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    X = df.drop(columns=['Survived', 'PassengerId'])
    y = df['Survived']
    return df, X, y

def import_train_data_withcabins():
    df = pd.read_csv('data/train2.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    X = df.drop(columns=['Survived', 'PassengerId'])
    y = df['Survived']
    return df, X, y

def import_clean_test():
    df = pd.read_csv('src/test.csv')
    df.Age = df.Age.fillna(df.Age.mean())
    df.Fare = df.Fare.fillna(df.Fare.mean())
    embarked_dummies = pd.get_dummies(df['Embarked'], dtype=int)
    df = pd.concat([df.drop(columns=['Embarked']), embarked_dummies], axis=1)
    sex_dummies = pd.get_dummies(df['Sex'], drop_first=True, dtype=int)
    df = pd.concat([df.drop(columns=['Sex']), sex_dummies], axis=1)
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    return df

def import_clean_test_withcabins():
    df = pd.read_csv('src/test.csv')
    
    df.Age = df.Age.fillna(df.Age.mean())
    df.Fare = df.Fare.fillna(df.Fare.mean())
    
    cabin_letters = []
    for cabin in df.Cabin:
        if pd.isna(cabin):
            cabin_letters.append("Unknown")
        else:
            cabin_letters.append(cabin[0])
    df['cabin_letters'] = cabin_letters
    df.drop(columns=['Cabin'], inplace=True)
    cabin_dummies = pd.get_dummies(df['cabin_letters'], prefix='cabin', dtype=int)
    df = pd.concat([df.drop(columns=['cabin_letters']), cabin_dummies], axis=1)
    df['cabin_T'] = 0
    cols = list(df.columns)
    cols.remove("cabin_T")
    idx = cols.index("cabin_Unknown")
    cols.insert(idx, "cabin_T")
    df = df[cols]
    
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='embarked', dtype=int)
    df = pd.concat([df.drop(columns=['Embarked']), embarked_dummies], axis=1)
    
    sex_dummies = pd.get_dummies(df['Sex'], drop_first=True, dtype=int)
    df = pd.concat([df.drop(columns=['Sex']), sex_dummies], axis=1)
    
    df.drop(columns=['Name', 'Ticket'], inplace=True)
    return df

def generate_results(model, test_df, output_csv_name):
    y_pred = model.predict(test_df.drop(columns=['PassengerId']))
    results = pd.DataFrame({
    "PassengerId":test_df['PassengerId'],
    "Survived":y_pred
    })
    results.set_index('PassengerId', inplace=True)
    results.to_csv(f'results/{output_csv_name}')

def generate_scaled_results(model, scaler, test_df, output_csv_name):
    test_scaled = scaler.transform(test_df.drop(columns=['PassengerId']))
    y_pred = model.predict(test_scaled)
    results = pd.DataFrame({
    "PassengerId":test_df['PassengerId'],
    "Survived":y_pred
    })
    results.set_index('PassengerId', inplace=True)
    results.to_csv(f'results/{output_csv_name}')