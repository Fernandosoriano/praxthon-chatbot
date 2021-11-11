import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF

from random import shuffle
import yaml

Data = None   # por qué inicilaizas data como NONE???
with open("training\intents.yaml","r") as file:
    Data = yaml.safe_load(file)

final_data = []
data_obj = [[(doc,item) for doc in Data[item]] for item in Data]
for data_list in data_obj:
    final_data.extend(data_list) #que diferncia hay entre apend y extend?

shuffle(final_data) #desordenas la información  por qué se desordena  la info?? para que no haya un overfiting

train    = final_data[:int(len(final_data)*0.9)]  #enterrenamiento por qupe multiplicas  por 0.9???
validate = final_data[int(len(final_data)*0.9):]  #para validar
# con los : puntos haces un slicing de las listas el 0.9 es el porcentaje de la info que traes

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform([item[0] for item in train])

#   => "Hola, como estas ?" 
#       hola: 1
#       como: 1
#       estas:1
#       fruta:0
#       mañana:0

# transform a count matrix to a normalized tf-idf representation (tf-idf transformer)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#   => "Hola, como estas ?"
#       hola: .2
#       como: .2
#       estas:.2
#       fruta:0
#       mañana:0

knn = KNeighborsClassifier(n_neighbors=int(len(Data)*1.5) ) 
# desition = DecisionTreeClassifier(random_state=0) random state, es la semilla
# NN = MLPClassifier(hidden_layer_sizes=(100,50,25,),random_state=1, max_iter=300) red neuronal
# capas ocultas, en este caso hay tres capas, que van disminuyendo su cantidad de nodos como un embudo
# random_state inicializas con params aleatorios
# max_iter cantidad de veces que la red neuronal se repite
# naiveB = GaussianNB()


clf = knn.fit(X_train_tfidf,[item[1] for item in train]) #le pasas los datos de entremnamiento en bruto y un alista que corresponde a la clasificacion de cada uno de estos elementos de forma ordenada
# clf = desition.fit(X_train_tfidf,[item[1] for item in train])
# clf = NN.fit(X_train_tfidf,[item[1] for item in train])
# clf = naiveB.fit(X_train_tfidf,[item[1] for item in train])


# [([0.2,0.36],saludo)]  ejemplo  que´es el 0.2 y el otro numero

# como ubicas los puntos en el plano, cómo mides la distancia??

# building up feature vector of our input
X_new_counts = count_vect.transform([item[0] for item in train])
# We call transform instead of fit_transform because it's already been fit
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicting the category of our input text: Will give out number for category
predicted = clf.predict(X_new_tfidf)

for doc, category in zip([item for item in train], predicted):
    print('%r => %s' % (doc, category))

print('We got an accuracy of',np.mean(predicted == [item[1] for item in train])*100, '% over the train data.')

X_new_counts = count_vect.transform([item[0] for item in validate])
# We call transform instead of fit_transform because it's already been fit
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# predicting the category of our input text: Will give out number for category
predicted = clf.predict(X_new_tfidf)

for doc, category in zip([item for item in validate], predicted):
    print('%r => %s' % (doc, category))

print('We got an accuracy of',np.mean(predicted == [item[1] for item in validate])*100, '% over the test data.')

