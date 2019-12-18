import tensorflow as tf
import numpy as np
import pointer_net 
import time,os,sys,json,re,requests
from elasticsearch import Elasticsearch
from nltk.util import ngrams

es = Elasticsearch()

tf.app.flags.DEFINE_integer("batch_size", 1,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 200, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 10, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 128, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 2, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", True, "Forward Only.")
#tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_string("data_path", "./data/pointercandidatevectors1.json", "Training Data path.")
tf.app.flags.DEFINE_string("test_data_path", "./data/pointercandidatevectorstest1.json", "Test Data path.")
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
    

  def convert(self,vector):
    inputs = []
    enc_input_weights = []
    dec_input_weights = []
    maxlen = 0
    questioninputs = []
    for idx,word in enumerate(vector):
      questioninputs.append(word)
      #inputs.append(questioninputs)
      enc_input_len = len(vector) 
    for i in range(FLAGS.max_input_sequence_len-enc_input_len):
      questioninputs.append([0]*801)
    weight = np.zeros(FLAGS.max_input_sequence_len)
    weight[:enc_input_len]=1
    enc_input_weights.append(weight)
    inputs.append(questioninputs)
    self.inputs = np.stack(inputs)
    self.enc_input_weights = np.stack(enc_input_weights)
    print("Load inputs:            " +str(self.inputs.shape))
    print("Load enc_input_weights: " +str(self.enc_input_weights.shape))


  def build_model(self):
    with self.graph.as_default():
      # Build model
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
      # Prepare Summary writer
      #self.writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',self.sess.graph)
      # Try to get checkpoint
      ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
      print(ckpt, FLAGS.log_dir)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Load model parameters from %s" % ckpt.model_checkpoint_path)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
      #print("Created model with fresh parameters.")
      #self.sess.run(tf.global_variables_initializer())


  def eval(self, vector):
    """ Randomly get a batch of data and output predictions """  
    print("inside eval again")
    self.convert(vector)
    #inputs,enc_input_weights, outputs, dec_input_weights = self.get_batch()
    #print(inputs.shape, enc_input_weights.shape,self.sess)
    predicted_ids = self.model.step(self.sess, self.inputs, self.enc_input_weights)
    print(predicted_ids) 
#    print("="*20)
#    for i in range(FLAGS.batch_size):
#      print("* %dth sample target: %s" % (i,str(outputs[i,1:]-2)))
#      for predict in predicted_ids[i]:
#        print("prediction: "+str(predict))       
#    print("="*20)

  def run(self, vector):
    self.eval(vector)


def getembedding(enturl):
  entityurl = '<http://www.wikidata.org/entity/'+enturl[37:]+'>'
  res = es.search(index="wikidataembedsindex01", body={"query":{"term":{"key":{"value":entityurl}}}})
  try:
    embedding = [float(x) for x in res['hits']['hits'][0]['_source']['embedding']]
    return embedding
  except Exception as e:
    print(enturl,' not found')
    return None
  return None


def getvector(sentence):
  q = sentence
  q = re.sub("\s*\?", "", q.strip())
  candidatevectors = []
  #questionembedding
  r = requests.post("http://localhost:8887/ftwv",json={'chunks': [q]})
  questionembedding = r.json()[0]
  tokens = [token for token in q.split(" ") if token != ""]
  true = []
  false = []
  ngramarr = []
  for n in range(1,4):
    ngramwords = list(ngrams(tokens, n))
    for tup in ngramwords:
      ngramjoined = ' '.join(tup)
      ngramarr.append([ngramjoined,n])
      #word vector
  r = requests.post("http://localhost:8887/ftwv",json={'chunks': [x[0] for x in ngramarr]})
  wordvectors = r.json()
  for wordvector,ngramtup in zip(wordvectors,ngramarr):
    word = ngramtup[0]
    n = ngramtup[1]
    esresult = es.search(index="wikidataentitylabelindex01", body={"query":{"multi_match":{"query":word}},"size":10})
    esresults = esresult['hits']['hits']
    if len(esresults) > 0:
      for idx,esresult in enumerate(esresults):
        entityembedding = getembedding(esresult['_source']['uri'])
        if entityembedding and questionembedding and wordvector:
          true.append(entityembedding+questionembedding+wordvector+[idx])
  return true  

def main(_):
  entitylinker = EntityLinker(FLAGS.forward_only)
  s = "Who is the president of India ?"
  v = getvector(s)
  entitylinker.run(v)

if __name__ == "__main__":
  tf.app.run()
