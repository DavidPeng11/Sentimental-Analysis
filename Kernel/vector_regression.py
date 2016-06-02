import gensim
import codecs
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr


def MakeArray(input_list, seed_value,  model):
    output_x = []
    output_y = []

    for i in input_list:
        output_x.append(model[i])
        output_y.append(seed_value[i])
    output_x = np.array(output_x)
    output_y = np.array(output_y)

    return output_x, output_y


if __name__ == '__main__':

    model = gensim.models.Word2Vec.load("C:\Pycharm\Projects\Word2Vec\wiki.zh.text.model")
    print ('model loaded')

    seed_file = codecs.open("C:\Pycharm\Projects\Kernel\seed_sim.txt", 'r', 'utf8')
    print ('seed file loaded')

    model_list = list(model.vocab.keys())

    seed_word_valence = dict()
    # seed_word_arousal = dict()
    for line in seed_file:
        line = line.strip().split('\t', -1)
        word_line = line[0]
        valence_line = line[1]
        arousal_line = line[2]
        seed_word_valence[word_line] = float(valence_line)
        # seed_word_arousal[word_line] = float(arousal_line)

    match_list = filter(lambda a: a in model_list, seed_word_valence.keys())

    train_list, test_list = train_test_split(match_list, test_size=0.3, random_state=0)
    x_train, y_train = MakeArray(train_list, seed_word_valence, model)


    x_test, y_test = MakeArray(test_list, seed_word_valence, model)


    lr = LinearRegression()
    lr.fit(x_train, y_train)

    y_predict = lr.predict(x_test)

    rmse = math.sqrt(mean_squared_error(y_test, y_predict))
    print ('the RMSE is %s '% rmse)
    mae = mean_absolute_error(y_test, y_predict)
    print ('the MAE is ', mae)
    r = pearsonr(y_test, y_predict)[0]
    print ('the pearsonr is' , r)
