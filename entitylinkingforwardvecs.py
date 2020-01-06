import tensorflow as tf
import numpy as np
import pointer_net 
import time,os,sys,json,re,requests
from elasticsearch import Elasticsearch
from nltk.util import ngrams

es = Elasticsearch()
tf.app.flags.DEFINE_integer("batch_size", 100,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 5000, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 1000, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 128, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 1, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", True, "Forward Only.")
#tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "frequence to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS

FLAGS.log_dir = './log'

class EntityLinker(object):
  def __init__(self, forward_only):
    self.forward_only = forward_only
    self.graph = tf.Graph()
    with self.graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      self.sess = tf.Session(config=config)
    self.build_model()
    

  def build_model(self):
    with self.graph.as_default():
      self.model = pointer_net.PointerNet(batch_size=FLAGS.batch_size, 
                    max_input_sequence_len=FLAGS.max_input_sequence_len, 
                    max_output_sequence_len=FLAGS.max_output_sequence_len, 
                    rnn_size=FLAGS.rnn_size, 
                    attention_size=FLAGS.attention_size, 
                    num_layers=FLAGS.num_layers,
                    beam_width=FLAGS.beam_width, 
                    learning_rate=FLAGS.learning_rate, 
                    max_gradient_norm=FLAGS.max_gradient_norm, 
                    forward_only=self.forward_only)
      ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
      print(ckpt, FLAGS.log_dir)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Load model parameters from %s" % ckpt.model_checkpoint_path)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)


  def eval(self):
    """ Randomly get a batch of data and output predictions """  
    #print("inside eval again")
    chunks = (len(self.test_inputs) - 1) // FLAGS.batch_size + 1
    for i in range(chunks):
      test_input_batch = self.test_inputs[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
      test_enc_input_weights_batch = self.test_enc_input_weights[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
      gtbatch = self.outputs[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
      predicted_ids = self.model.step(self.sess, test_input_batch, test_enc_input_weights_batch)
      #print(predicted_ids, predicted_ids.shape) 
      #print(gtbatch, len(gtbatch))
      #print(len(self.test_inputs))
      return predicted_ids, gtbatch 

  def run(self):
    return self.eval()


  def getvector(self,d):
    inputs = []
    enc_input_weights = []
    dec_input_weights = []
    maxlen = 0
    self.outputs = []
    for question in d:
      questioninputs = []
      questionoutputs = []
      for idx,word in enumerate(question):
        #print(len(word[0]), word[1], word[2])
        questioninputs.append(word[0])
        if word[2] == 1.0:
          questionoutputs.append(idx+1)
      enc_input_len = len(question)
      #print(enc_input_len)
      #if len(questionoutputs) == 0:
      #    continue
      #inputs.append(questioninputs)
      for i in range(FLAGS.max_input_sequence_len-enc_input_len):
        questioninputs.append([0]*802)
      output=[]
      for i in questionoutputs:
        output.append(int(i))
      self.outputs.append(output)
      weight = np.zeros(FLAGS.max_input_sequence_len)
      weight[:enc_input_len]=1
      enc_input_weights.append(weight)
      inputs.append(questioninputs)
  
    self.test_inputs = np.stack(inputs)
    self.test_enc_input_weights = np.stack(enc_input_weights)
    #np.save('gt.npy',self.outputs)
    #np.save('test_inputs.npy',self.test_inputs)
    #np.save('test_enc_input_weights.npy',self.test_enc_input_weights)
        
    print("Load test inputs:            " +str(self.test_inputs.shape))
    print("Load test enc_input_weights: " +str(self.test_enc_input_weights.shape))


def calculatef1(batchd, predictions, groundtruths,tp,fp,fn):
  
  for inputquestion,prediction,groundtruth in zip(batchd, predictions, groundtruths):
    idtoentity = {}
    predents = set()
    gtents = set()
    for entnum in list(prediction[0]):
      if entnum <= 0:
        continue
      predents.add(inputquestion[entnum-1][1])
    for entnum in groundtruth:
      if entnum <= 0:
        continue
      gtents.add(inputquestion[entnum-1][1])
    for goldentity in gtents:
      #totalentchunks += 1
      if goldentity in predents:
        tp += 1
      else:
        fn += 1
    for queryentity in predents:
      if queryentity not in gtents:
        fp += 1

  precisionentity = tp/float(tp+fp)
  recallentity = tp/float(tp+fn)
  f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
  print("precision entity = ",precisionentity)
  print("recall entity = ",recallentity)
  print("f1 entity = ",f1entity) 
  return tp,fp,fn

def main(_):
  entitylinker = EntityLinker(FLAGS.forward_only)
  tp = 0
  fp = 0
  fn = 0
#  if os.path.isfile('gt.npy') and os.path.isfile('test_inputs.npy') and os.path.isfile('test_enc_input_weights.npy'):
#    print("found numpy dumps, loading")
#    entitylinker.test_inputs = np.load('test_inputs.npy', allow_pickle=True)
#    entitylinker.test_enc_input_weights = np.load('test_enc_input_weights.npy', allow_pickle=True)
#    entitylinker.outputs = np.load('gt.npy', allow_pickle=True)
#    entitylinker.run()
#  else:

  #d = json.loads(open('data/pointercandidatevectorstest1.json').read())
  linecount = 0
  batchd = []
  with open('pointercandidatevectorstestfull1.json') as rfp:
    for line in rfp:
      line = line.strip()
      linecount += 1
      d = json.loads(line)
#      #print(len(d))
      batchd.append(d)
      if linecount % FLAGS.batch_size == 0:
        print("process batch")
        print(linecount)
        try:
          entitylinker.getvector(batchd)
        except Exception as e:
          print(e)
          continue
        predicted, groundtruth = entitylinker.run()
        _tp,_fp,_fn = calculatef1(batchd,predicted,groundtruth,tp,fp,fn)
        tp = _tp
        fp = _fp
        fn = _fn
        batchd = []
    

if __name__ == "__main__":
  tf.app.run()
