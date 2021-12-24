import sys
import getopt
import os
import math
import operator
import pandas as pd
import numpy as np
import time

class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test. 
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is '1' or '0' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
#    self.stopList = set(self.readFile('./english.stop'))
    self.numFolds = 10

    self.posDict = dict()
    self.negDict = dict()
    self.posCount = 0
    self.negCount = 0

    self.alpha = 1
  #  self.vocabSize = 0

  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  #
  # If any one of the FILTER_STOP_WORDS and BOOLEAN_NB flags is on, the
  # other one is meant to be off.

  def classify(self, words):
    """ TODO
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    if self.FILTER_STOP_WORDS:
      words =  self.filterStopWords(words)
    
    # Write code here
    posWordsCount = sum(self.posDict.values())
    negWordsCount = sum(self.negDict.values())
    vocabSize = len(self.posDict) + len(self.negDict)

    
    posProb = 0.0
    negProb = 0.0
    for word in words:
#      posProb *= (self.posDict.get(word,0) + 1) * 1.0 / (posWordsCount + vocabSize + 1)
#      negProb *= (self.negDict.get(word,0) + 1) * 1.0 / (negWordsCount + vocabSize + 1)
      posProb = posProb + math.log(self.posDict.get(word,0) + self.alpha) - math.log(posWordsCount + self.alpha * vocabSize + 1)
      negProb = negProb + math.log(self.negDict.get(word,0) + self.alpha) - math.log(negWordsCount + self.alpha * vocabSize + 1)

#    print(self.posCount)
#    print(self.negCount)
    posProb = posProb + math.log(self.posCount * 1.0 / (self.posCount + self.negCount))
    negProb = negProb + math.log(self.negCount * 1.0 / (self.posCount + self.negCount))
#    print("posProb: %s", posProb)
#    print("negProb: %s", negProb)
    if posProb >= negProb:
      return '1'
    else:
      return '0'
  

  def addExample(self, klass, words):
    """
     * TODO
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier 
     * in the NaiveBayes class.
     * Returns nothing
    """
    
#    beforeLen = len(words)

    if self.BOOLEAN_NB:
      words = list(set(words))

#    afterLen = len(words)

#    if afterLen < beforeLen:
#      print("words before set: %s", beforeLen)
#      print("words after set: %s", afterLen)

#    else:
    
    if klass == '1':
      self.posCount += 1
      for word in words:
        self.posDict.update({word : self.posDict.get(word,0) + 1})
    else:
      self.negCount += 1
      for word in words:
        self.negDict.update({word : self.negDict.get(word,0) + 1})

    # Write code here

    pass
      

  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainFile):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    self.train_df = pd.read_csv(trainFile, usecols=['text','humor_controversy'])
    for idx, row in self.train_df.iterrows():
        if not np.isnan(row['humor_controversy']):
            example = self.Example()
            example.words = self.segmentWords(row['text'])
            example.klass = str(int(row['humor_controversy']))
            split.train.append(example)

#    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
#    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
#    for fileName in posTrainFileNames:
#      example = self.Example()
#      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
#      example.klass = 'pos'
#      split.train.append(example)
#    for fileName in negTrainFileNames:
#      example = self.Example()
#      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
#      example.klass = 'neg'
#      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)

  def generate_fold_list(self, l, n):
    """ l is the list to be chunked, n is the chunk size"""
    for i in range(0, len(l), n):
        yield l[i : i + n]

  def crossValidationSplits(self, train_file):
    """Returns a list of TrainSplits corresponding to the cross validation splits."""
    splits = [] 

    # first read in all the train csv file into 2 lists
    example_list = []
    self.train_df = pd.read_csv(train_file, usecols=['text','humor_controversy'])

    for idx, row in self.train_df.iterrows():
        if not np.isnan(row['humor_controversy']):
            example = self.Example()
            example.words = self.segmentWords(row['text'])
            example.klass = str(int(row['humor_controversy']))
#            print(type(example.klass))
#            print(example.klass)
            example_list.append(example)
    
    chunk_size = math.ceil(len(example_list) / self.numFolds)
    split_list = list(self.generate_fold_list(example_list, chunk_size))
#    print(len(split_humor_list))
#    for a in split_humor_list:
#        print(len(a))
    
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()

      split.test = split_list[fold]
#      print(len(humor_split.test))

      concatenated_list = split_list[0:fold] + split_list[fold+1:]
      for l in concatenated_list:
        split.train += l;
#      humor_split.train =       
#      print(len(humor_split.train))
      
      splits.append(split)

    return splits
  
  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB):

  nb = NaiveBayes()
  splits = nb.crossValidationSplits(args[0])

  avgAccuracy = 0.0
  fold = 0
  for split in splits:
#    print('split.train size: %s' % len(split.train))
#    print('split.test size: %s' % len(split.test))
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    accuracy = 0.0
    for example in split.train:
      words = example.words
      classifier.addExample(example.klass, words)
  
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0
  
    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)) 
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB, trainFile, evalFile, testFile = None):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  trainSplit = classifier.trainSplit(trainFile)
  start_train_time = time.time()
  classifier.train(trainSplit)
  total_train_time = time.time() - start_train_time
  evalSplit = classifier.trainSplit(evalFile)
#  accuracy = 0.0

  tp = 0.0
  tn = 0.0
  fp = 0.0
  fn = 0.0

  total_eval_time = 0.0
  for example in evalSplit.train:
    words = example.words
    start_eval_time = time.time()
    guess = classifier.classify(words)
    total_eval_time += time.time() - start_eval_time
    if example.klass == guess:
#      accuracy += 1.0
        if guess == '1':                                                                         
            tp += 1.0
        else:
            tn += 1.0
    else:
        if guess == '1':
            fp += 1.0
        else:
            fn += 1.0
  accuracy = (tp+tn) / (tp+tn+fp+fn)
  print('[INFO]\tAccuracy: %f' % accuracy)
  precision = (tp) / (tp+fp)
  print('[INFO]\tPrecision: %f' % precision)
  recall = (tp) / (tp+fn)
  print('[INFO]\tRecall: %f' % recall)
  f1 = 2*(recall*precision) / (recall + precision)
  print('[inFO]\tF1: %f' % f1)
  print('[INFO]\tTrain Time: %s seconds' % total_train_time)
  print('[INFO]\tDev Time %s seconds' % total_eval_time)

#  accuracy = accuracy / len(testSplit.train)
#  print('[INFO]\tAccuracy: %f' % accuracy)

  if testFile:
    guesses = []
    testSplit = classifier.trainSplit(testFile)
#    print(len(testSplit.train))                                                                 
    for example in testSplit.train:
      words = example.words
      guess = classifier.classify(words)
      guesses.append(guess)
    fw = open('test1c_result.txt', 'w')
    print(len(guesses))
    for guess in guesses:
      fw.write(guess+'\n')
    fw.close()


def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  
  if len(args) == 2:
#    print(args)
    classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB,  args[0], args[1])
  elif len(args) == 1:
#    print(args)
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB)
  elif len(args) == 3:
    classifyDir(FILTER_STOP_WORDS, BOOLEAN_NB,  args[0], args[1], args[2])
    

if __name__ == "__main__":
    main()
