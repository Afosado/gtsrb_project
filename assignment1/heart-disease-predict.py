import pandas as pd
import numpy as np

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# metrics
from sklearn.metrics import accuracy_score, log_loss

# classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# global variables
dataset_file_path = './processed.cleveland.data'

def load_data(filename):
    '''
    Reads specified .csv file and returns an X and y dataframe.
    > 0. age
    > 1. sex
    > 2. chest pain type(4 values)
    > 3. resting blood pressure
    > 4. serum cholestoral in mg/dl
    > 5. fasting blood sugar > 120 mg/dl
    > 6. resting electrocardiographic results(values 0, 1, 2)
    > 7. maximum heart rate achieved
    > 8. exercise induced angina
    > 9. oldpeak = ST depression induced by exercise relative to rest
    > 10. the slope of the peak exercise ST segment
    > 11. number of major vessels(0-3) colored by flourosopy
    > 12. thal: 3 = normal, 6 = fixed defect, 7 = reversable defect
    > 13. num: 0 = no presence, 4 = present
    '''

    # reading the data
    try:
        print("Reading .csv")
        data = pd.read_csv(filename, header=None)
        print("Finished reading .csv")
    except:
        print("Unable to read .csv")

    # set column names
    attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    data.columns = attributes

    X, y = data.iloc[:, 0:-1], data.iloc[:, -1]

    return X, y

def preprocess_data(data):
    '''
    Arguments: Pandas Dataframe (X_train or X_test)
    Return: Preprocessed np array
    '''
    # saving columns and indices since ColumnTransformer removes them
    columns = data.columns
    index = data.index

    # defining categorical and numerical features (and categorical feature value range)
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 
                             'exang', 'slope', 'ca', 'thal']
    categories = [[0,1], [1,2,4], [0,1], [0,1,2], 
                  [0,1], [1,2,3], [0,1,2,3], [3,6,7]]
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # creating transformers
    # categorical_transformer = Pipeline[('onehot', OneHotEncoder())]
    # numerical_transformer = Pipeline[('scaler', StandardScaler())]

    # creating and applying ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numerical_features),
                      ('cat', OneHotEncoder(categories=categories, 
                                            handle_unknown='ignore'),
                       categorical_features)],
        n_jobs=-1)

    data = preprocessor.fit_transform(data)
    
    return data


def main():
    # loading data
    X, y = load_data(dataset_file_path)

    # creating train/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, 
                                                        shuffle=True)

    X_train = preprocess_data(X_train)
    X_test = preprocess_data(X_test)

    # converting to np arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    assert X_train.shape[1] == X_test.shape[1], 'Num features in X_train differs from X_test' 
    assert X_train.shape[0] == y_train.shape[0], 'Number of examples does not match for X_train and Y_train'
    assert X_test.shape[0] == y_test.shape[0], 'Number of examples does not match for X_test and Y_test'

    # list of classifiers to train
    classifiers = [
        KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        SVC(kernel="rbf", gamma='scale', C=0.025, probability=True),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_jobs=-1),
        AdaBoostClassifier(),
        GradientBoostingClassifier()
    ]

    for clf in classifiers:
        clf.fit(X_train, y_train)
        print(clf)
        print(clf.score(X_test, y_test))
        print()

if __name__ == "__main__":
    main()
