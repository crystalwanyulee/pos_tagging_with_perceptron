<p style='LINE-HEIGHT:20px'> </p>
<div style="TEXT-ALIGN:center; FONT-SIZE:24px"><b>POS tagging with the Perceptron algorithm</b></div>

<p style='LINE-HEIGHT:6px'> </p>

#### 1. Code Structure

There are 4 files:

* `data_structures.py` can produce a set of features for each position in a sentence, including the previous tag, the current word, the previous two words, the following two words, prefix (the first three letters), and suffix (the last three letters).
* `perceptron_pos_tagger.py` includes the learning and inference part in the perceptron algorithm and the averaged perceptron algorithm.  
  * `train` conducts online training, decoding, and weights-updating through the Perceptron algorithm. Here we can set `mode='sum'` or `mode='avg'` to run a normal Perceptron model or an averaged Perceptron model.
  * `tag` does pos-tagging.
  * `complie_data` initializes weights and stored in a dictionary before training.  
  * `decoder` implements Viterbi algorithm
  * `accuracy` is used for computing accuracy on the development data.
* `train_test_tagger.py` includes data-reading, model-building, and all kind of experiments. 

<p style='LINE-HEIGHT:6px'> </p>

#### 2. Experimental Settings and Results

All experiments are run in the `train_test_tagger.py`. The results are presented in the following table:



<img src="image-20201105153449441.png" alt="image-20201105153449441" style="zoom:80%;" />



In order to further examine the results, I visualized them to see if their is any significant difference in different features sets and two kind of models.

<div style="page-break-after: always; break-after: page;"></div>
<p style='LINE-HEIGHT:20px'> </p>





* **Perceptron Model**
  
  * The best performing features set is the group without previous words, while the poorest one is the combination without prefix and suffix. The range is about 2.7% (0.9383-0.9112 = 0.0271).
  * Taking out some features, including previous words and previous tags, can slightly improve the Perceptron model's performance.
  
  

![All Models 1](All Models 1.png)





* **Averaged Perceptron Model**
  
  * The model has the best performance when including all features. Conversely, the model performs worst when removing prefix and suffix features. This outcome is the same as the one in the original Perceptron model. The difference between the best and the worst models is about 2.7% (0.9390-0.9065 = 0.0295), which is slightly higher than the one in the Perceptron model.
  * Excluding any features cannot help the performance become better.
  
  

![All Models 2](All Models 2.png)

<div style="page-break-after: always; break-after: page;"></div>
<p style='LINE-HEIGHT:20px'> </p>





* **Perceptron Model vs. Averaged Perceptron Model**
  
  * In the four feature groups, the original Perceptron model performs better than the Averaged Perceptron model. 
  * When removing the most influential feature group, prefix and suffix, accuracy in the Averaged Perceptron model decreases more obviously, from 93.90% to 90.65%, while the one in the Perceptron model goes down from 93.57% to 91.12%.
  
  

![All Models 3](All Models 3.png)





#### 3. Insights and Final Thoughts

Basically, the Perceptron model and the Averaged Perceptron model have similar results, although the latter has a broader range between different feature sets. The ablation study shows that prefix, suffix, and next two words are more influential in POS tagging. The results indeed make sense because we can infer the part-of-speech of a word from their first or last several letters. Although their POS probably change due to different contexts, we can further determine them through their following words.  However, the previous word and previous tag seem not to have significant impacts on prediction. We can conclude that the following words can bring more helpful information for the model than previous words.





#### 4. Plain Data Name

* auto_dev_data.tagged

* auto_test_data.tagged

