import gensim
import codecs
import numpy as np
import math
# # regression model
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.svm import SVR
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

model = gensim.models.Word2Vec.load("C:\Pycharm\Projects\Word2Vec\wiki.zh.text.model")

print('model loaded')
vocab_list = list(model.vocab.keys())

seed_file = codecs.open("C:\Pycharm\Projects\Kernel\seed_sim.txt", 'r', 'utf8')

print ('seed file loaded')
known_word_valence = dict()
# known_word_arousal = dict()
for line in seed_file:
    line1, line2 = line.split('\t', 2)[:-1]
    known_word_valence[line1] = float(line2)
    # known_word_arousal[line1] = float(line3)

K_list = filter(lambda a: a in vocab_list, known_word_valence.keys())
# Find the words which are match with the corpus
Train_list, Test_list = train_test_split(K_list, test_size=0.3, random_state=0)
# N_list=K_list[:1000]
# # get the first 1000 words as the training set
# L_list=K_list[1000:]

x_train = []
for i in Train_list:
    x_train_row = [1]
    for j in Train_list:
        d = model.similarity(i, j)
        # f=math.sqrt(d)
        # f=math.exp(d)
        f=math.log(d)
        v = known_word_valence[j]
        x_train_row.append(f * v)
    x_train.append(list(x_train_row))

x_train = np.array(x_train)
print ('x_train shape:', x_train.shape)

y_train = []
for w in Train_list:
    v = known_word_valence[w]
    y_train.append(v)

y_train = np.array(y_train)
print ('y_trains shape: ', y_train.shape)

x_test = []
for i in Test_list:
    x_test_row = [1]
    for j in Train_list:
        d = model.similarity(i, j)
        # f=math.sqrt(d)
        # f=math.exp(d)
        f=math.log(d)
        v = known_word_valence[j]
        x_test_row.append(f * v)
    x_test.append(list(x_test_row))

x_test = np.array(x_test)
print ('x_test shape:', x_test.shape)

y_test = []
for w in Test_list:
    v = known_word_valence[w]
    y_test.append(v)
y_test = np.array(y_test)
print ('y_test shape:', y_test.shape)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

rmse = math.sqrt(mean_squared_error(y_test, y_pred))
print rmse
mae = mean_absolute_error(y_test, y_pred)
print mae
r = pearsonr(y_test, y_pred)[0]
print r
pass
