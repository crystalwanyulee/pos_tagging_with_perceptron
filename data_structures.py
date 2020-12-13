class Sentence(object):
    def __init__(self, sent, label=True):
        ''' Modify if necessary.
        '''
        self.sent = sent
        self.label = label
        
        
        if self.label:
            self.words = [i for i, j in self.sent]
            self.tags = [j for i, j in self.sent]
            
            
        else:
            self.words = [i for i in self.sent]
            self.tags= None


    def features(self, position):
        ''' Implement your feature extraction code here. This takes annotated or unannotated sentence
        and return a set of features
        '''
        
        cur_word = self.words[position]
        
        if self.label:
            cur_pos = self.tags[position]
        else:
            cur_pos = None
        
        prefix = cur_word[:3]
        suffix = cur_word[-3:]
        

        if position == 0:
            pre_pos1 = 'START_'
            pre_word2 = 'START__'
            pre_word1 = 'START_'
            
        else:
            if position == 1:
                pre_word2 = 'START_'
            else:
                pre_word2 = self.words[position-2]
            
            if self.label:
                pre_pos1 = self.tags[position-1]
            else:
                pre_pos1 = None
                
            pre_word1 = self.words[position-1]      
            
        if position == len(self.sent)-1:
            
            next_word1 = 'END_'
            next_word2 = 'END__'

        else:
            
            if position == len(self.sent)-2:
                next_word2 = 'END_'
                
            else:
                next_word2 = self.words[position+2]
                
            next_word1 = self.words[position+1]
            
    
        #feature_set = {'cur_pos': cur_pos, 
                       #'features': [pre_pos1, cur_word, pre_word1, pre_word2, next_word1, next_word2, prefix, suffix]}
        
        feature_set = {'cur_pos': cur_pos, 'pre_pos': pre_pos1, 'cur_word': cur_word,
                       'pre_word1': pre_word1, 'pre_word2': pre_word2, 'next_word1': next_word1, 
                       'next_word2': next_word2, 'prefix': prefix, 'suffix': suffix}
            
        return feature_set




