# part one

import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv(r'ehrdata.csv')

with open(r'ehrdata.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)

    for row in reader:
        print(row)
categorical_columns = ['SOURCE', 'SEX', 'AGE']
label_encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])


def genresults():
    y = df.SEX
    X = df.drop('SEX', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    reg = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    ACC = (accuracy_score(y_pred, y_test), 0.5)
    CR = classification_report(y_pred, y_test)
    return reg, y_pred, ACC, CR


def gencoef():
    return genresults()[0].coef_


class LogModel:
    def __init__(self):
        pass


def main():
    cof = gencoef()
    print(cof)


if __name__ == '__main__':
    main()


from phe import paillier
import json


# Create your views here.
def collectkeys():
    public_key, private_key = paillier.generate_paillier_keypair()
    keys = {}
    keys['public_key'] = {'n': public_key.n}
    keys['private_key'] = {'p': private_key.p, 'q': private_key.q}
    with open('hospkeys.json', 'w') as file:
        json.dump(keys, file)


def obtainkeys():
    with open('hospkeys.json', 'r') as file:
        keys = json.load(file)
        pub_key = paillier.PaillierPublicKey(n=int(keys['public_key']['n']))
        priv_key = paillier.PaillierPrivateKey(pub_key, keys['private_key']['p'], keys['private_key']['q'])
        return pub_key, priv_key


def serializedata(public_key, labdata):
    encrypted_data_list = [public_key.encrypt(x) for x in labdata]
    encrypted_data = {}
    encrypted_data['public_key'] = {'n': public_key.n}
    encrypted_data['values'] = [(str(x.ciphertext()), x.exponent) for x in encrypted_data_list]
    serialized = json.dumps(encrypted_data)
    return serialized


pub_key, priv_key = obtainkeys()
labdata = HAEMATOCRIT, HAEMOGLOBINS, ERYTHROCYTE, LEUCOCYTE, THROMBOCYTE, MCH, MCHC, AGE, SOURCE, MCV = [43.5, 14.8, 5.39,
                                                                                                      12.7, 334, 27.5,
                                                                                                      34.0, 0, 1, 80.7]
serializedata(pub_key, labdata)
datafile = serializedata(pub_key, labdata)
with open('labdata.json', 'w') as file:
    json.dump(datafile, file)

