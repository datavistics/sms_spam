# Goal
Given this [dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection), I want to beat the Matthews Correlation Coefficient (MCC) of `0.893` that was found in the 
[dataset paper](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/doceng11.pdf). The MCC metric is useful 
because it performs well with imbalanced data like ours (much more Ham than Spam). This result was considered state
of the art in 2011.
# Approach
I have already written a fair amount of detail in my two notebooks - Deep Learning and Model Comparison. 
This report will duplicate some of that information for consolidation.

When approaching the SMS Spam classification I thought it should have a flow of **test-train-split** (TTS) -> 
**tokenization** -> **model** -> **prediction**. After that it was about determining the best ways to execute
each step, and then fine-tuning. I chose 3 models to try: SVM, Naive Bayes (NB), and Random Forest (RF). I expected 
SVM and NB to do very well, and was unsure about RF.

## Test-Train-Split
With a TTS, you want to find a good balance. You want as much data as you can to train, but you simultaneously
need as much data as you can to predict on. If you have too little training data, you will find your model to be
underfit. Conversely if you have too much training data you will find that its an easy job to predict, but when
you deploy your model it wont generalize well and your results have been skewed.

I chose a TTS of 80-20 for my training and hyperparameter search.

## Tokenization
There were 2 tokenizations used in the dataset paper, I found them both to be very useful due to the nature of
our data. CRM114 simplified preserves many of the features that for non-spam tasks we might consider noise.
These features help us *fingerprint* the spam. It keeps the odd spellings and punctuation that might appear 
as those can be strong indicators of spam. The second tokenization used a more *eager* splitting algorithm as 
it would split on some punctuation.

## Model Training
In training the models, I used Grid Search Cross Validation which allows me to search over many hyper-parameter
possibilities very efficiently. I chose a linear kernel for the SVM, as most text tokenizations similar to mine 
are linearly separable. It was suggested in the dataset paper to investigate n-grams, and this was helpful on the
MNB, but the other classifiers did not find them helpful.

Surprising to me, MNB and RF found the eager split algorithm to be more effective, where the SVM found the 
simplified CRM114 to be better. I believe the eager worlds well because more splitting drives a less sparse 
feature set.

# Room for improvement
## Feature Engineering
I believe that more features will help predict spam more effectively. I recommend looking at letter 
capitalizations, number of punctuation per SMS, misspelled words.

## Deep Learning
Kaggle [hosted](https://www.kaggle.com/uciml/sms-spam-collection-dataset) this dataset and there was a fair
amount of interest. In particular this [notebook](https://www.kaggle.com/jacklinggu/keras-mlp-cnn-test-for-text-classification)
caught my eye. There weren't very good model efficacy metrics so I rewrote some of it, but left the architecture
in tact.

The results were promising, I found an MCC of `0.9391`. I believe with some feature engineering and fine-tuning
this result could be better than what I achieved with the MNB. It warrants investigation.

# Conclusion
My goal was to beat the old MCC result, and I did so with a good margin. This is a testament to how far Machine
Learning and NLP have come in the last 6-7 years. I think Deep Learning is the future, but I was very excited to
see such a simple algorithm like MNB perform extremely well. If the results are equal, simpler is desired. It is
faster, smaller, easier to work with, and easier to debug.