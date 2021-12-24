import random
import sys
import getopt
import csv
import time


class Perceptron:
  class TrainSplit:
    def __init__(self):
      self.train = []
      self.test = []
      self.dev = []

  class Example:
    def __init__(self):
      self.klass = ''
      self.words = []

  def __init__(self):
    self.numFolds = 10
    self.doc_word_freq = {}
    self.weights = {}
    self.bias = 0.0
    self.weight_final = {}
    self.b_final = 0.0
    self.weight_avg = {}
    self.b_avg = 0.0
    self.c = 1

    for i in range(len(self.weights)):
      self.weights[i] = random.random() * 0.99 + .01

  def classify(self, words):
    word_dict = {}
    result = self.b_final

    for word in words:
      if word in word_dict:
        word_dict[word] += 1
      else:
        word_dict[word] = 1

    for (key, freq) in word_dict.items():
        if key not in self.weight_final:
            self.weight_final[key] = 0.0
        result += self.weight_final[key] * freq

    if result > 0:
        return 'pos'
    else:
        return 'neg'

  def addExample(self, klass, words):
    self.doc_word_freq = {}

    for word in words:
      if word in self.doc_word_freq:
        self.doc_word_freq[word] += 1
      else:
        self.doc_word_freq[word] = 1
    self.c += 1
    pass

  def train(self, split, iterations):
      random.shuffle(split.train)

      for i in range(iterations):
        for example in split.train:
            words = example.words
            self.addExample(example.klass, words)
            result = self.bias

            for (key, freq) in self.doc_word_freq.items():
              if key not in self.weights:
                self.weights[key] = 0.0
              result += self.weights[key] * freq

            perceptron_output = 0.0
            if result > 0:
              perceptron_output = 1.0

            target_value = 0.0
            if example.klass == 'pos':
              target_value = 1.0

            if target_value != perceptron_output:
              for (key, freq) in self.doc_word_freq.items():
                self.weights[key] += (target_value - perceptron_output) * float(freq)
                if key not in self.weight_avg:
                  self.weight_avg[key] = 0.0
                self.weight_avg[key] += self.c * (target_value - perceptron_output) * float(freq)
              self.bias += (target_value - perceptron_output)
              self.b_avg += self.c * (target_value - perceptron_output)

      for (key, weight) in self.weights.items():
        if key not in self.weight_avg:
          self.weight_avg[key] = 0.0
        self.weight_final[key] = self.weights[key] - self.weight_avg[key] / self.c

      self.b_final = self.bias - self.b_avg / self.c

  def readFile(self, fileName):
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents))
    return result


  def segmentWords(self, s):
    return s.split()


  def trainSplit(self, trainDir):
    split = self.TrainSplit()
    with open(trainDir, 'r') as read_obj:
      csv_reader = csv.reader(read_obj)
      header = next(csv_reader)
      if header != None:
        for row in csv_reader:
          print(row)
          example = self.Example()
          example.words = row[1].split()
          if row[2] is "0":
            example.klass = 'neg'
          else:
            example.klass = 'pos'
          split.train.append(example)
    return split

  def makeSplit(self, trainDir, devDir, predict_type):
    split = self.TrainSplit()
    with open(trainDir, 'r') as read_obj:
      csv_reader = csv.reader(read_obj)
      header = next(csv_reader)
      if header != None:
        for row in csv_reader:
          example = self.Example()
          example.words = row[1].split()
          example_class = ""
          if predict_type is "isHumor":
            example_class = row[2]
          elif predict_type is "Controversy":
            if row[2] is '0':
              continue
            example_class = row[4]

          if example_class is "0":
            example.klass = 'neg'
          else:
            example.klass = 'pos'
          split.train.append(example)

    with open(devDir, 'r') as read_obj:
      csv_reader = csv.reader(read_obj)
      header = next(csv_reader)
      if header != None:
        for row in csv_reader:
          example = self.Example()
          example.words = row[1].split()
          example_class = ""
          if predict_type is "isHumor":
            example_class = row[2]
          elif predict_type is "Controversy":
            if row[2] is '0':
              continue
            example_class = row[4]

          if example_class is "0":
            example.klass = 'neg'
          else:
            example.klass = 'pos'
          split.dev.append(example)
    return split

def classifyDir(args):
  pt_is_humor = getModel(args, "isHumor")
  pt_co = getModel(args, "Controversy")

  # Get results for test1000.csv
  testFile = args[2]
  print("Geting results...")
  results = {}
  with open(testFile, 'r') as read_obj:
    csv_reader = csv.reader(read_obj)
    header = next(csv_reader)
    if header is not None:
      for row in csv_reader:
        id = row[0]
        words = row[1].split()
        result = []
        guess_humor = pt_is_humor.classify(words)
        guess_co = pt_co.classify(words)
        if guess_humor is 'neg':
          result.append('0') # is_humor
          result.append('0') # cotroversy
        else:
          result.append('1') # is_humor
          if guess_co is 'neg': # cotroversy
            result.append('0')
          else:
            result.append('1')
        results[id] = result

  with open('percepetron_result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "is_humor", "humor_rating","humor_controversy","offense_rating"])
    for key in results:
      writer.writerow([key, results[key][0], "0", results[key][1], "0"])
  print("Done")

def getModel(args, predict_type):
  print("Task 1a: Training for", predict_type)
  pt = Perceptron()
  split = pt.makeSplit(trainDir=args[0], devDir=args[1], predict_type=predict_type)
  print("Training data amount: ", len(split.train))

  iterations = int(args[3])
  tp = 0.0
  fp = 0.0
  fn = 0.0
  begin_t = time.time()
  pt.train(split, iterations)
  end_t = time.time()
  print("Training time (in seconds): ", end_t - begin_t)

  print("Dev data amount: ", len(split.dev))
  for example in split.dev:
    words = example.words
    guess = pt.classify(words)
    if example.klass == guess:
      tp += 1.0
    else:
      if guess is '1':
        fp += 1.0
      else:
        fn += 1.0
  t2 = time.time()
  print("Predicting time (in seconds): ", t2 - end_t)
  acc = tp / len(split.dev)
  f1 = tp / (tp + 0.5 * fp + 0.5 * fn)

  print('[INFO]\tTask 1a: %s Accuracy: %f' % (predict_type, acc))
  print('[INFO]\tTask 1a: %s  F-1 socre: %f \n' % (predict_type, f1))
  return pt

def main():
  (options, args) = getopt.getopt(sys.argv[1:], '')
  classifyDir(args)

if __name__ == "__main__":
    main()
