import pandas as pd

def import_train_data():
    df = pd.read_csv('data/train1.csv')
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