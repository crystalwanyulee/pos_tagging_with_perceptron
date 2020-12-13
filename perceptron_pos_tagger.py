
from collections import defaultdict
import numpy as np
from time import time
from copy import copy

class Perceptron_POS_Tagger(object):
    def __init__(self,):
        ''' Modify if necessary. 
        '''
        self.weights = defaultdict(dict)
        self.n_pos = 0 
        self.idtopos = {}
        self.postoid = {}
        self.fname = ['pre_pos', 'cur_word', 'pre_word1', 'pre_word2', 
                      'next_word1', 'next_word2', 'prefix', 'suffix']
        self.m = defaultdict(dict)


    def tag(self, test_data, tag_only=False):
        ''' Implement the Viterbi decoding algorithm here.
        '''
        test_pred = []
        
        for sent in test_data:
            length_sent = len(sent.sent)
            b = np.array([], dtype=int)
            
            for i in range(length_sent):
        
                f_dict = sent.features(i)            
                features = [f_dict[f] for f in self.fname]               
                b = np.append(b, self.decoder(f_dict)) 
            
            if tag_only:
                test_pred.append(b)
            else:
                test_pred.append([[sent.words[i], pos] for i, pos in enumerate(b)])
                
        return test_pred
        


    def train(self, train_data, dev_data, mode='sum', remove_f=-1):
        ''' Implement the Perceptron training algorithm here.
        '''

        start = time()

        # The default is to use all features 
        if remove_f != -1:
            self.fname = [f for f in self.fname if f not in remove_f]   
            
        if len(self.weights) == 0:
            self.compile_data(train_data)
        
        if mode == 'avg':
            self.m = copy(self.weights)

        # Online Training and Decoding
        for sent in train_data:
            
            length_sent = len(sent.sent)
            correct_seq = []
            f_set = [] 
            
            b = np.array([], dtype=int)
            
            for i in range(length_sent):
                
                f_dict = sent.features(i)
                correct_seq.append(self.postoid[f_dict['cur_pos']])
                
                features = [f_dict[f] for f in self.fname]
                f_set.append(features)                 # return pos id
                b = np.append(b, self.decoder(f_dict, pos_name=False))
                
            
            # Updating weights
            update = {'corr': 1, 'err': -1}
            seq = {'corr': correct_seq , 'err': b} 
            mis_id = np.where(b != correct_seq)[0]
            
            if len(mis_id) == 0:
                continue
            
            
            if mode == 'sum':
            
                mis_pos = [(i, seq[j][i], update[j]) for j in ['corr', 'err'] for i in mis_id]
                for i, name in enumerate(self.fname):
                    if name == 'pre_pos':
                        for j, pos_id, val in mis_pos:
                            if j != (length_sent-1):
                                self.weights[name][self.idtopos[pos_id]][correct_seq[j+1]] += val
                                
                            if j == 0:
                                pre_pos = 'START_'
                                
                            else:
                                pre_pos = self.idtopos[correct_seq[j-1]]
                                
                            self.weights[name][pre_pos][pos_id] += val
                            
                    else:
                        for j, pos_id, val in mis_pos:
                            self.weights[name][f_set[j][i]][pos_id] += val

            
            if mode == 'avg':           
                    
                seq_t = np.append(mis_id, length_sent)
                avg_weight = [seq_t[i]-seq_t[i-1] for i in range(1,len(seq_t))]
                mis_pos = [(i, seq[j][i], update[j], avg_weight[k]) for j in ['corr', 'err'] for k, i in enumerate(mis_id)]
            
                for i, name in enumerate(self.fname):
                    if name == 'pre_pos':
                        for j, pos_id, val, w in mis_pos:
                            if j != (length_sent-1):
                                self.weights[name][self.idtopos[pos_id]][correct_seq[j+1]] += val
                                self.m[name][self.idtopos[pos_id]][correct_seq[j+1]] += val*(w/length_sent)
                                
                            if j == 0:
                                pre_pos = 'START_'
                                
                            else:
                                pre_pos = self.idtopos[correct_seq[j-1]]
                                
                            self.weights[name][pre_pos][pos_id] += val
                            self.m[name][pre_pos][pos_id] += val*(w/length_sent)
                            
                    else:
                        for j, pos_id, val, w in mis_pos:
                            self.weights[name][f_set[j][i]][pos_id] += val
                            self.m[name][f_set[j][i]][pos_id] += val*(w/length_sent)
                        


        # Evaluating performance
        dev_auto = self.tag(dev_data, tag_only=True)
        dev_gold = [list(tuple(zip(*sent.sent))[1]) for sent in dev_data]
        dev_acc = self.accuracy(dev_gold, dev_auto)
        
        end = time()
        
        print('Feature Set: {0}\nDev Accuracy: {1:.4f}\nTime: {2:.2f}\n'.format(self.fname, dev_acc, end-start))
        
        return dev_acc


    def compile_data(self, train_data):
        
        vocab, pos = [], []
        
        for sent in train_data:
            word, p = tuple(zip(*sent.sent))
            vocab.extend(word)
            pos.extend(p)
        
        vocab, pos = set(vocab), set(pos)
        
        prefix = set([word[:3] for word in vocab])
        suffix = set([word[-3:] for word in vocab])
        
        self.n_pos = len(pos)
        self.idtopos = {i: p for i, p in enumerate(pos)}
        self.postoid  = {v:k for k, v in self.idtopos.items()}
    

        for name in self.fname:
            
            if name == 'pre_pos':
                  self.weights['pre_pos'] = {p:np.zeros(self.n_pos) for p in ['START_', 'UNKNOWN_'] + list(pos)}
                  
            elif name == 'prefix':
                prefix = set([word[:3] for word in vocab])
                self.weights['prefix'] = {p:np.zeros(self.n_pos) for p in list(prefix) + ['UNKNOWN_']}
                
            elif name == 'suffix':
                suffix = set([word[-3:] for word in vocab])
                self.weights['suffix'] = {s:np.zeros(self.n_pos) for s in list(suffix) + ['UNKNOWN_']}
        
            else:
                value_list = ['START__', 'START_', 'END_', 'END__', 'UNKNOWN_']
                self.weights[name] = {word:np.zeros(self.n_pos) for word in value_list + list(vocab)}   
                

    def decoder(self, f_dict, pos_name=True):
        
        score = np.zeros(self.n_pos)
        
        for name in self.fname:
            feature = f_dict[name]
            
            if feature not in self.weights[name].keys():
               feature = 'UNKNOWN_' 
               
            if (feature == '.') & (name == 'pre_pos'):
                continue
            
            score += self.weights[name][feature] 

        b = score.argmax()
        
        if pos_name:
            b = self.idtopos[b]

        return b

    
    def accuracy(self, gold, auto):
        correct = 0.0
        total = 0.0
        for g_snt, a_snt in zip(gold, auto):
            correct += sum([g_tup == a_tup for g_tup, a_tup in zip(g_snt, a_snt)])
            total += len(g_snt)
        
        return correct/total

       


