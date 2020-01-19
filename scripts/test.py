import tensorflow as tf
import numpy as np
import pointer_net 
import time,os,sys,json,re,requests
from elasticsearch import Elasticsearch
from nltk.util import ngrams

es = Elasticsearch()
tf.app.flags.DEFINE_integer("max_input_sequence_len", 3000, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 100, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 512, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 500, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 1, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_string("test_data", "./a.txt", "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", True, "Forward Only.")
tf.app.flags.DEFINE_string("models_dir", "./log", "Log directory")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "frequence to do per checkpoint.")
tf.app.flags.DEFINE_integer("batch_size", 10, "batchsize")

FLAGS = tf.app.flags.FLAGS

#FLAGS.log_dir = './lcquad2logs4'


class EntityLinker(object):
  def __init__(self, forward_only):
    self.forward_only = forward_only
    self.graph = tf.Graph()
    with self.graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.operation_timeout_in_ms=6000
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
      ckpt = tf.train.get_checkpoint_state(FLAGS.models_dir)
      print(ckpt, FLAGS.models_dir)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Load model parameters from %s" % ckpt.model_checkpoint_path)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
      self.sess.graph.finalize()


  def eval(self):
    """ Randomly get a batch of data and output predictions """ 
    predicted_ids,outputs = self.model.step(self.sess, self.test_inputs, self.test_enc_input_weights, update=False)
    print(outputs)
    return predicted_ids,outputs 

  def run(self):
    return self.eval()


  def getvector(self,d):
    inputs = []
    self.outputs = []
    enc_input_weights = []
    dec_input_weights = []
    maxlen = 0
    self.outputs = []
    for question in d:
      questioninputs = []
      enc_input_len = len(question[2])
      print(enc_input_len)
      if enc_input_len > FLAGS.max_input_sequence_len:
        print("Length too long, skip")
        continue
      for idx,word in enumerate(question[2]):
        questioninputs.append(word[0])
      for i in range(FLAGS.max_input_sequence_len-enc_input_len):
        questioninputs.append([0]*803)
      self.outputs.append(question[1])
      weight = np.zeros(FLAGS.max_input_sequence_len)
      weight[:enc_input_len]=1
      enc_input_weights.append(weight)
      inputs.append(questioninputs)
    self.test_inputs = np.stack(inputs)
    self.test_enc_input_weights = np.stack(enc_input_weights)


  def calculatef1(self, batchd, predictions, decoderoutput, tp,fp,fn):
    for inputquestion,prediction,groundtruth in zip(batchd, predictions, self.outputs):
      idtoentity = {}
      predents = set()
      gtents = groundtruth
      #print(len(self.test_inputs))
      seen = []
      for entnum in list(prediction[0]):
        if entnum <= 0:
          continue
        wordindex = inputquestion[2][entnum-1][0][801]
        ngramtype = inputquestion[2][entnum-1][0][802]
        print(inputquestion[2][entnum-1][0][801], inputquestion[2][entnum-1][0][802],inputquestion[2][entnum-1][0][800], inputquestion[2][entnum-1][1])
        if wordindex in seen:
            continue
#        if ngramtype == -2 and wordindex-1 in seen:
#            seen.append(wordindex)
#            continue
        predents.add(inputquestion[2][entnum-1][1])
        seen.append(wordindex)
#      for entnum in groundtruth:
#        if entnum <= 0:
#          continue
#        gtents.add(inputquestion[entnum-1][1])
      print("scores ",decoderoutput[1].scores)
      print(gtents,predents)
      for goldentity in gtents:
        #totalentchunks += 1
        if not goldentity:
          print("Skip None")
          continue
        if goldentity in predents:
          tp += 1
        else:
          fn += 1
      for queryentity in predents:
        if queryentity not in gtents:
          fp += 1
    try: 
      precisionentity = tp/float(tp+fp)
      recallentity = tp/float(tp+fn)
      f1entity = 2*(precisionentity*recallentity)/(precisionentity+recallentity)
      print("precision entity = ",precisionentity)
      print("recall entity = ",recallentity)
      print("f1 entity = ",f1entity) 
    except Exception as e:
      print(e)
    print(tp,fp,fn)
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
  with open(FLAGS.test_data) as rfp:
    for line in rfp:
      line = line.strip()
      d = json.loads(line)
      linecount += 1
      if len(d) > FLAGS.max_input_sequence_len:
        print("Skip question, too long")
        continue
      batchd.append(d)
      print(linecount)
      if len(batchd) == FLAGS.batch_size:
        try:
          entitylinker.getvector(batchd)
          predicted,decoderoutput = entitylinker.run()
          _tp,_fp,_fn = entitylinker.calculatef1(batchd,predicted,decoderoutput,tp,fp,fn)
          tp = _tp
          fp = _fp
          fn = _fn
          batchd = []
        except Exception as e:
          print(e)
          batchd = []
          continue
    

if __name__ == "__main__":
  tf.app.run()
