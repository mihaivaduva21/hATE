from keras.models import model_from_json
from keras.preprocessing import sequence
import pickle
import pandas as pd
from nltk.stem.porter import *

stemmer = PorterStemmer()

def load_tokenizer(fname):
    with open(fname, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def load_model(model_fname, weights_fname):
    json_file = open(model_fname, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_fname)
    loaded_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return loaded_model

def predict_comment(comment):

    comment = " ".join([stemmer.stem(w) for w in comment.split()])
    comment = [comment]
    tokenized = tokenizer.texts_to_sequences(comment)
    padded = sequence.pad_sequences(tokenized, 1000)
    pred = model.predict(padded)[0][0]
    print(" :" , comment,  " pred :" , pred)
    return pred

def load_hate(path='./data/train.csv'):
    #read the csv data
    df = pd.read_csv(path)
    #print(df) 
    #extract the image pixels
    X_train = df.drop(columns = ['label'])

    X_train['comment'] = X_train['comment'].apply(lambda row : predict_comment(row))
    print(X_train)
    #extract the labels
    y_train = df['label'].values

    return X_train , y_train

def to_txt (path='./data/train.csv'):

    f1 = open("./data/nohate.txt",'w')
    f2 = open("./data/hate.txt",'w')
    df = pd.read_csv(path)

    for row in  df.itertuples():
            if (row.label == "nohate"):
                f1.write(row.comment+"\n")
            else:
                f2.write(row.comment+"\n")
    print(df)

    
model = load_model('model.json', 'model.h5')
tokenizer = load_tokenizer('tokenizer.pickle')

to_txt()
load_hate()

