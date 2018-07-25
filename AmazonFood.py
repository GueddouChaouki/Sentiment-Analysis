# -*- coding: utf-8 -*-
"""

GUEDDOU CHAOUKI

"""
import time
start_time1 = time.time()
from keras.utils.np_utils import to_categorical
#from keras import regularizers
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
amazon_food = pd.read_csv('E:\matser 2\Memeoire\Nouveau dossier\Amazon_food.csv', encoding='utf-8')
#clean data 
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def clean_data(amazo):
    
    sup_links = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", amazo)
    
    sup_hashtags = re.sub("#[A-Za-z0-9]+", " " , sup_links)
    
    sup_balis = re.sub("<[^>]+ >"," ",sup_hashtags)
    print("lin supp .....")
    only_letters = re.sub("[^a-zA-Z]", " ", sup_balis)
    print("tokenization .....")
    tokens = nltk.word_tokenize(only_letters)
    print("lower  .....")
    lower_case = [l.lower() for l in tokens]
    print("stop words .....")
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    print(" racinisation .....")
    stemmed = [stemmer.stem(item) for item in filtered_result]
    print("lematisaton .....")
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in stemmed]
    print("terminer")
    return lemmas

# (" ".join(lemmas))
#applique sur le jeux de données amazon_food 
print('******clean data***')
print("links supp .....")
print("hashtags supp .....")
print("balise supp .....")
print("tokenization .....")
print("lower  .....")
print("stop words .....")
print(" racinisation .....")
print("lematisaton .....")

amazon_food['Text_clean'] = amazon_food['Text'].apply(clean_data)
amazon_food[['Text','Text_clean']].head()

# normalisation le score
print('******normalisation le score ***')
def partition(x):
    if x > 3:
        return "posi"
    if x==3:
        return "neu"
    if x < 3:
        return "nega"
    
amazon_food['Score']=amazon_food['Score'].apply(partition)
Score = amazon_food['Score']
text_clean_rev1 = amazon_food['Text_clean']
text_clean_rev2 = amazon_food['Text_clean']


from gensim.models import Word2Vec
from keras.preprocessing import sequence
#from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense , Activation
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing.text import Tokenizer 
n=200
start_time = time.time()



text_vector_w2v= getAvgFeatureVecs(text_clean_rev1, w2v, 200)
print("---Time taken is %s seconds ---" % (time.time() - start_time))



#********************model deeplearning**********************************************     
print("lstm...")         
w2v.save("w2v_200features")
embedding_matrix_w2v = w2v.wv.syn0
print("Shape of embedding matrix : ", embedding_matrix_w2v.shape)
nbr_words = embedding_matrix_w2v.shape[0] 
tokenizer = Tokenizer(nb_words=nbr_words) 
tokenizer.fit_on_texts(text_clean_rev2)
w2v = Word2Vec(text_clean_rev1, size =n )
X_train2, X_test2, y_train2, y_test2 = train_test_split(text_clean_rev2, Score, test_size=0.2, random_state=42)
#review to integer 
sequences_train = tokenizer.texts_to_sequences(X_train2)
sequences_test = tokenizer.texts_to_sequences(X_test2)
review_length = amazon_food["Text_clean"].dropna().map(lambda x: len(x))
plt.figure(figsize=(12,8))
review_length.loc[review_length < 1500].hist()
plt.title("Distribution of Review Length aprés filtrage")
plt.xlabel('Review length ')
plt.ylabel('Count')
pd.DataFrame(review_length).describe()
review_length = amazon_food["Reviews_clean"].dropna().map(lambda x: len(x))
plt.figure(figsize=(12,8))
review_length.loc[review_length < 1500].hist()
plt.title("Distribution of Review Length aprés filltrage")
plt.xlabel('Review length ')
plt.ylabel('Count')
pd.DataFrame(review_length).describe()

#aprés les statstique sur la taille  de review limiter aux 100 premiers mots donc maxlen = 100
maxlen = 200 
X_train_seq = sequence.pad_sequences(sequences_train, maxlen=maxlen)
X_test_seq = sequence.pad_sequences(sequences_test, maxlen=maxlen)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_le = le.fit_transform(y_train2)
y_test_le = le.transform(y_test2)
y_train_oh = to_categorical(y_train_le)
y_test_oh = to_categorical(y_test_le)
print('X_train shape:', X_train_seq.shape)
print('X_test shape:', X_test_seq.shape) 
print('y_train shape:', y_train_oh.shape) 
print('y_test shape:', y_test_oh.shape)


batch_size = 100
nb_classes = 3
nb_epoch = 10
neurons = 100               
activation_function = 'softmax'  
loss = 'categorical_crossentropy'                  
optimizer="adam"
#***********************************LSTM********************************************
print( " ***** buid modele LSTM*****")
def build_model_lstm(embedding_matrix , nb_classes, neurons, activ_func=activation_function, loss=loss, optimizer=optimizer):
  """
  """
  embedding_layer = Embedding(embedding_matrix.shape[0], 
                         embedding_matrix.shape[1],input_length= maxlen ,
                          weights=[embedding_matrix])
  model2 = Sequential()
  model2.add(embedding_layer)
  model2.add(LSTM(neurons , return_sequences=True , activation='tanh'))
  model2.add(LSTM(neurons, activation='tanh' ))
  model2.add(Dense(nb_classes ))
  model2.add(Activation(activ_func))
  model2.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  model2.summary()
  return model2
                          
#********* build model lstm **********
LSTM_model = build_model_lstm(embedding_matrix_w2v, nb_classes=nb_classes, neurons=neurons)
                 #****** train model on data ***********p
print("  train model LSTM")
start_time = time.time()
lstm = LSTM_model.fit(X_train_seq, y_train_oh, epochs=nb_epoch, batch_size=batch_size, verbose=1)
print("---Time taken is %s seconds ---" % (time.time() - start_time))
print('X_train shape:', X_train_seq.shape)
print('X_test shape:', X_test_seq.shape)
print('y_train shape:', y_train_oh.shape) 
print('y_test shape:', y_test_oh.shape)
print('nbr de classes = ',nb_classes)
print('*************parameters**********')             
print('nomber de poche = :',nb_epoch) 
print('batch size  =', batch_size) 
print('fonction activation', activation_function)
print('loss =',loss)
print('optimizer = ',optimizer)
print('nbre de hidden layers LSTM =',2)
print ('nbr neurons ',neurons)                 
print('********* evaluation model***************')
            #*********evaluation model******
score = LSTM_model.evaluate(X_test_seq, y_test_oh, batch_size=28, verbose=1)
#print('loss_test = ' ,score[0])
print('acc_test',score[1])
acc_test_lstm = score[1]
print ("accuracy de teset: ",acc_test_lstm)

print("result LSTM trian loss et train acc")
train_acc = lstm.history['acc']
train_loss = lstm.history['loss']
plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='Train loss')
plt.plot(range(1, len(train_acc)+1), train_acc, color='red', label='train acc')
plt.xlim(0, len(train_loss))
plt.legend(loc="upper right")
plt.xlabel('nbre Epoch')
plt.ylabel('Loss & acc')
plt.show()

print("comparaison....")

print("---Time taken is %s seconds ---" % (time.time() - start_time1))

from keras.models import model_from_json
model_josn =LSTM_model.to_json()
with open("model.json","w") as json_file :
    json_file.write(model_josn)
LSTM_model.save_weights("model.h5")
print("saved model to disk ")

X_train1, X_test1, y_train1, y_test1 = train_test_split(text_vector_w2v, Score, test_size=0.2, random_state=42)

#********************************* DecisionTree ***********************************************************
from sklearn.tree import DecisionTreeClassifier
start_time = time.time()
print("DecisionTree..")
from sklearn.metrics import accuracy_score
start_time = time.time()
clf_DTC = DecisionTreeClassifier()
clf_DTC.fit(X_train1, y_train1)
y_pred_DTC = clf_DTC.predict(X_test1)
acc_DTC = accuracy_score(y_test1, y_pred_DTC)
print (" Decision tree accuracy: ",acc_DTC)
print("---Time taken is %s seconds ---" % (time.time() - start_time))

#*********************************SVM**************************************************************
start_time = time.time()
from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
clf_svm.fit(X_train1, y_train1)
y_pred_svm = clf_svm.predict(X_test1)
acc_svm = accuracy_score(y_test1, y_pred_svm)
print( "Linear SVM accuracy: ",acc_svm )
print("---Time taken is %s seconds ---" % (time.time() - start_time)) 

#*********************************Random Forest**************************************
print("Random Forest..")
from sklearn.ensemble import RandomForestClassifier
start_time = time.time()
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train1, y_train1)
y_pred_rf = clf_rf.predict(X_test1)
acc_rf = accuracy_score(y_test1, y_pred_rf)
print ("random forest accuracy: ",acc_rf )
print("---Time taken is %s seconds ---" % (time.time() - start_time))

