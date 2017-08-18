#Albert bot framework
import gensim
import pandas as pd
import csv
import sys
import json
import numpy as np
import nn_lib
from scipy import spatial
import math
import sklearn.decomposition as decomp
import mailMe
import theano

class Albert:

    def __init__(self):
        #need to do some data preprocessing
        #self.loadGoogleVectors()
        self.word_map = None
        self.common_words = None

    def loadGoogleVectors(self):
        model = gensim.models.KeyedVectors.load_word2vec_format('../GoogleWord2Vec/GoogleNews-vectors-negative300.bin', binary=True)
        model.save_word2vec_format('googlenews.txt')

    def readInMyFile(self, my_file):
        #run models with mostly training, and have only one dev set due to limited data
        q_a_list = []
        punctuation = [",", "?", ".", "!", "(", ")"]
        df = pd.read_csv(my_file)
        questions = df['questions'].tolist()
        answers = df['answers'].tolist()
        for question, answer in zip(questions, answers):
            question = question.split(' ')
            answer = answer.split(' ')
            question_new = []
            answer_new = []
            #need to print q_word and a_word separately
            for q_word in question:
                #find character to replace, there is only one possible punct per word
                cq_replacement = None
                for character_q in q_word:
                    for punct in punctuation:
                        if (character_q==punct):
                            cq_replacement = punct
                #replace the appropriate words
                if cq_replacement is not None:
                    question_new.append(q_word.replace(cq_replacement, ""))
                    question_new.append(cq_replacement)
                else:
                    question_new.append(q_word)
            for a_word in answer:
                #find character to replace, there is only one possible punct per word
                aq_replacement = None
                for character_a in a_word:
                    for punct in punctuation:
                        if (character_a==punct):
                            aq_replacement = punct
                #replace the appropriate words
                if aq_replacement is not None:
                    answer_new.append(a_word.replace(aq_replacement, ""))
                    answer_new.append(aq_replacement)
                else:
                    answer_new.append(a_word)
            q_a_list.append([question_new, answer_new])
        return q_a_list
    def generateInputWordMap(self):
        #upload first 1000 words from the google corpus
        num_common_words = 1000
        json_map = {}
        with open('googlenews.txt') as googleVec:
            googleVec.next()
            for i in xrange(num_common_words):
                line = googleVec.next()
                line = line.replace("\n", "")
                vector = line.split(' ')
                word_w_vec = vector[0]
                rep_vector = vector[1:]
                #replace string with float
                rep_vector = [float(value) for value in rep_vector]
                json_map[unicode(word_w_vec, errors='ignore')] = rep_vector
        with open('googleCommon.txt', 'w+') as googleCommon:
            googleCommon.write(json.dumps(json_map))
        self.pca_wordMap("googleCommon.txt", "googleCommon_reduced.txt", 38)
        print "completed generating the common words"

    def obtainCommonWords(self):
        with open('googleCommon_reduced.txt') as googleCommon:
            self.common_words = json.loads(googleCommon.readlines()[0])

    def generateWordMap(self, my_file):
        #create json word map for given training data
        q_a_list = self.readInMyFile(my_file)
        word_list = []
        json_map = {}
        #create a word list for all words
        for pair in q_a_list:
            for q_a in pair:
                for q_a_word in q_a:
                    if q_a_word.lower() not in word_list:
                        word_list.append(q_a_word.lower())
        #loop through all the words
        for word, i in zip(word_list, xrange(len(word_list))):
            word = unicode(word, errors='ignore')
            print "looking for word \"" + word + "\""
            with open('googlenews.txt') as googleVec:
                #get rid of first input
                googleVec.next()
                found = False
                while(not found):
                    try:
                        line = googleVec.next()
                        line = line.replace("\n", "")
                        vector = line.split(' ')
                        word_w_vec = vector[0]
                        if word_w_vec.lower() == word.lower():
                            found = True
                            rep_vector = vector[1:]
                            #replace string with float
                            rep_vector = [float(value) for value in rep_vector]
                            #add to the map
                            json_map[word] = rep_vector
                        
                    except StopIteration:
                        #add word
                        found = True
                        print "word " + word + " not found"
                        json_map[word] = np.random.uniform(-1,1,300).tolist()
            print str(i) + "/" + str(len(word_list))


        #print json to file
        with open('my_json_wordMap.txt', 'w+') as wordMap:
            wordMap.write(json.dumps(json_map))
        print "completed writing new json word map file"
    
    def getWordMap(self, my_file):
        with open(my_file) as word_map:
            return json.loads(word_map.readlines()[0])

    def encode_x_y(self, word_map_file, train_data_file, vector_length, lstm_chain_length):
        q_a_list = self.readInMyFile(train_data_file)
        self.word_map = self.getWordMap(word_map_file)
        #list of q a vector sentence pairs
        q_number = []
        a_number = []
        for pair in q_a_list:
            q_sentence = []
            a_sentence = []
            for q_word in pair[0]:
                q_sentence.append(self.word_map[unicode(q_word.lower(), errors='ignore')])
            for a_word in pair[1]:
                a_sentence.append(self.word_map[unicode(a_word.lower(), errors='ignore')])
            # if q_number and a_number are not 40 long then append zeros
            while len(q_sentence) < lstm_chain_length:
                q_sentence.append(np.zeros(vector_length).tolist())
            while len(a_sentence) < lstm_chain_length:
                a_sentence.append(np.zeros(vector_length).tolist())
            #now add the new sentences to q and a
            q_number.append(q_sentence)
            a_number.append(a_sentence)
        q_number = np.array(q_number)
        a_number = np.array(a_number)
        a_number = a_number.reshape(a_number.shape[0], a_number.shape[1]*a_number.shape[2])
        print q_number.shape
        print a_number.shape
        return q_number.tolist(), a_number.tolist()

    def get_max_length(self, train_data_file):
        data = self.readInMyFile(train_data_file)
        max_len = 0
        max_sentence = None
        for pair in data:
            for sentence in pair:
                if (len(sentence) > max_len):
                    max_len = len(sentence)
                    max_sentence = sentence
        print max_len
        return

    def getOutput(self, output):
        min_cosine_dist = 2.0*math.pi
        outputSentence = []
        outputString = ""
        #split the output in form [[val val val val]] to [[val], [val], [val], [val]]
        splitOutput = list(self.chunks(output[0], 38))
        for word in splitOutput:
            #look for closest word in word map
            word_key = None
            for key in self.word_map.iterkeys():
                dist = 1 - spatial.distance.cosine(word, self.word_map[key])
                if (dist<min_cosine_dist):
                    min_cosine_dist = dist
                    word_key = key
            outputSentence.append(word_key)
        for word in outputSentence:
            if word is not None:
                outputString = outputString + " " + word
        print outputString

    def transform_input(self, user_input, word_map_file):
        #get the word map
        self.word_map = self.getWordMap(word_map_file)
        #get the common words
        self.obtainCommonWords()
        #split the input
        user_input = user_input.split(' ')
        #declare empty x_in
        x_in = []
        for word in user_input:
            if word in self.word_map:
                x_in.append(self.word_map[word])
            elif word in self.common_words:
                x_in.append(self.common_words[word])
            else:
                #then replace the word with a random 38 long word vector
                rand_word = np.random.uniform(0.0, 1.0, np.array([1, 38])).tolist()[0]
                x_in.append(rand_word)
        #make the sentence 38 "words" long
        if (len(x_in)>38):
            x_in = x_in[:38]
        while len(x_in) is not 38:
            x_in.append(np.zeros(38).tolist())
        return x_in
            

    def chunks(self, l, n):
        n = max(1, n)
        return (l[i:i+n] for i in xrange(0, len(l), n))

    def pca_wordMap(self, word_map_file, word_map_file_output, vector_length):
        word_map = self.getWordMap(word_map_file)
        word_map_new = {}
        dataset_collection = []
        for value in word_map.itervalues():
            dataset_collection.append(value)
        pca = decomp.PCA(n_components=vector_length)
        pca.fit(dataset_collection)
        for key in word_map.iterkeys():
            word_map_new[key] = pca.transform(word_map[key]).tolist()[0]
        with open(word_map_file_output, 'w+')as output:
            output.write(json.dumps(word_map_new))
        return

def albertAttemptOne():
    #python for post training
    albert = Albert()
    reply = None
    user_input = None
    w_list = []
    b_list = []
    u_list = []
    model_information = "modeltrain0.txt"
    weight_files = ['modeltrain9_w_list.save', 'modeltrain9_b_list.save', 'modeltrain9_u_list.save']
    weight_names = [w_list, b_list, u_list]
    network = nn_lib.loadModel(model_information, weight_files, weight_names, 1, 38)
    print "loading the bot function created..."
    my_func = network.use_function()
    while(True):
        #get the input
        if reply is None:
            user_input = raw_input("HiIiYa I'm a CrAZY BotT:\n\n")
        else:
            user_input = raw_input(reply + "\n\n")
        #get the x_in
        x_in = albert.transform_input(user_input, 'my_json_wordMap_reduction.txt')
        output = my_func(*x_in)
        reply = albert.getOutput(output)


if __name__ == "__main__":
    albertAttemptOne()
    try:
        mailMe.sendEmail("albert training starting with learning rate " + str(1.0))
        lstm_chain_length = 38
        vector_length = 38
        dim_s_e = (vector_length*lstm_chain_length)

        albert = Albert()
        #albert.pca_wordMap('my_json_wordMap.txt', 'my_json_wordMap_reduction.txt', vector_length)
        albert.get_max_length('training_data.csv')
        x_in, y_out = albert.encode_x_y('my_json_wordMap_reduction.txt', 'training_data.csv', vector_length, lstm_chain_length)
        
        #build the model
        dimensions = [[dim_s_e, 100], [100, dim_s_e]]
        #print the network function
        print "compile network"
        network = nn_lib.Neural_network(1.0, 'float64')
        #add lstm
        print "compile lstm chain"
        my_lstm_chain = network.lstm_chain(vector_length, lstm_chain_length, 'float64')
        #add neural network
        print "compile nn"
        network.fully_connected_network(1, dimensions, my_lstm_chain)
        #return function
        print "compile whole func"
        my_func = network.return_compiled_func()
        epochs = 100
        with open('loss_pattern.txt', 'w+') as loss_file:
            for i in xrange(epochs):
                for sample_x, sample_y in zip(x_in, y_out):
                    print "my_func executing"
                    my_func(*(sample_x + [[sample_y]]))
                    print "loss file writing"
                    loss_value = network.print_loss(sample_x + [[sample_y]])
                    loss_file.write(str(loss_value))
                network.saveModel('modeltrain' + str(i))
                mailMe.sendEmail("The latest model has been saved at epoch " + str(i) + " and loss of " + str(loss_value) + " at lr of 1.0")
                loss_file.write("saving model at loss above")
    except (Exception):
        mailMe.sendEmail("albert err out :/")