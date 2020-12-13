import sys
import runpy
import os
os.chdir('C://Users/admin/Documents/Statistical Approches of NLP/cs134-2020-project3/cs134-2020-project3/project3')
from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence
from time import time


def read_in_gold_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [Sentence(line) for line in lines]

    return sents 


def read_in_plain_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        sents = [Sentence(line, label=False) for line in lines]

    return sents 


def output_auto_data(auto_data, filename):
    ''' According to the data structure you used for "auto_data",
        write code here to output your auto tagged data into a file,
        using the same format as the provided gold data (i.e. word_pos word_pos ...). 
    '''
    
    results = []
    
    for data in auto_data:
    
        results.append(' '.join(['_'.join(i) for i in data]))
    

    with open(filename, "w") as f:
        f.write('\n'.join(results))
    
    


if __name__ == '__main__':

    saved_argv = sys.argv
    sys.argv = ['', 'train/ptb_02-21.tagged', 'dev/ptb_22.tagged', 'dev/ptb_22.snt', 'test/ptb_23.snt'] 

    # Run python train_test_tagger.py train/ptb_02-21.tagged dev/ptb_22.tagged dev/ptb_22.snt test/ptb_23.snt to train & test your tagger
    train_file = sys.argv[1]
    gold_dev_file = sys.argv[2]
    plain_dev_file = sys.argv[3]
    test_file = sys.argv[4]

    # Read in data
    train_data = read_in_gold_data(train_file)
    gold_dev_data = read_in_gold_data(gold_dev_file)
    plain_dev_data = read_in_plain_data(plain_dev_file)
    test_data = read_in_plain_data(test_file)

    dev_acc = {'sum': [], 'avg': []}
    mname = {'sum': 'Perceptron', 'avg': 'Averaged Perceptron'}
    
    
    # Conduct Experiments
    for mode in ['sum', 'avg']:
        
        # Experiment 1: All features
        print(mname[mode] + ' ' + 'Experiment 1: All features')
        my_tagger = Perceptron_POS_Tagger()
        dev_acc[mode].append(my_tagger.train(train_data, gold_dev_data, mode=mode))
    
        # Experiment 2: Removing Current Word
        print(mname[mode] + ' ' + 'Experiment 2: Removing Current Word')
        my_tagger = Perceptron_POS_Tagger()
        dev_acc[mode].append(my_tagger.train(train_data, gold_dev_data, mode=mode, remove_f=['cur_word']))
    
        # Experiment 3: Removing Previous Two Words
        print(mname[mode] + ' ' + 'Experiment 3: Removing Previous Two Words')
        my_tagger = Perceptron_POS_Tagger()
        dev_acc[mode].append(my_tagger.train(train_data, gold_dev_data, mode=mode, remove_f=['pre_word1', 'pre_word2']))
        
        # Experiment 4: Removing Next Two Words
        print(mname[mode] + ' ' + 'Experiment 4: Removing Next Two Words')
        my_tagger = Perceptron_POS_Tagger()
        dev_acc[mode].append(my_tagger.train(train_data, gold_dev_data, mode=mode, remove_f=['next_word1', 'next_word2']))
    
        # Experiment 5: Removing Prefix and Suffix
        print(mname[mode] + ' ' + 'Experiment 5: Removing Prefix and Suffix')
        my_tagger = Perceptron_POS_Tagger()
        dev_acc[mode].append(my_tagger.train(train_data, gold_dev_data, mode=mode, remove_f=['prefix', 'suffix']))
    
        # Experiment 6: Removing Previous Tag
        print(mname[mode] + ' ' + 'Experiment 6: Removing Previous Tag')
        my_tagger = Perceptron_POS_Tagger()
        dev_acc[mode].append(my_tagger.train(train_data, gold_dev_data, mode=mode, remove_f=['pre_pos'])) 


    # Apply your tagger on dev & test data
    auto_dev_data = my_tagger.tag(plain_dev_data)
    auto_test_data = my_tagger.tag(test_data)


    # Outpur your auto tagged data
    output_auto_data(auto_dev_data, 'auto_dev_data.tagged')
    output_auto_data(auto_test_data, 'auto_test_data.tagged')


   # Run Scorer
    sys.argv = ['', 'dev/ptb_22.tagged', 'auto_dev_data.tagged']
    runpy.run_path('scorer.py', run_name='__main__') 
    sys.argv = saved_argv
