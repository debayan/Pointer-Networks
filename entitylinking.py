import tensorflow as tf
import numpy as np
import pointer_net 
import time
import os
import sys
import json

tf.app.flags.DEFINE_integer("batch_size", 128,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 200, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 10, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 128, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 128, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 4, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 2, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_string("data_path", "./data/pointercandidatevectors1.json", "Training Data path.")
tf.app.flags.DEFINE_string("test_data_path", "./data/pointercandidatevectorstest1.json", "Test Data path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200, "frequence to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS

class EntityLinker(object):
  def __init__(self, forward_only):
    self.forward_only = forward_only
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.sess = tf.Session()
    self.read_data()
    self.read_test_data()
    self.build_model()
    

  def read_test_data(self): 
    d = json.loads(open(FLAGS.test_data_path,'r').read())
    inputs = []
    enc_input_weights = []
    outputs = []
    dec_input_weights = []
    maxlen = 0
    for question in d:
      questioninputs = []
      questionoutputs = []
      for idx,word in enumerate(question):
        questioninputs.append(word[0])
        if word[2] == 1.0:
          questionoutputs.append(idx+1)
      #inputs.append(questioninputs)
      enc_input_len = len(question) 
      for i in range(FLAGS.max_input_sequence_len-enc_input_len):
        questioninputs.append([0]*801)
      weight = np.zeros(FLAGS.max_input_sequence_len)
      weight[:enc_input_len]=1
      enc_input_weights.append(weight)
      inputs.append(questioninputs)

      output=[pointer_net.START_ID]
      for i in questionoutputs:
        # Add 2 to value due to the sepcial tokens
        output.append(int(i)+2)
      output.append(pointer_net.END_ID)
      dec_input_len = len(output)-1
      output += [pointer_net.PAD_ID]*(FLAGS.max_output_sequence_len-dec_input_len)
      output = np.array(output)
      outputs.append(output)
      weight = np.zeros(FLAGS.max_output_sequence_len)
      weight[:dec_input_len]=1
      dec_input_weights.append(weight)
        
    self.test_inputs = np.stack(inputs)
    self.test_enc_input_weights = np.stack(enc_input_weights)
    self.test_outputs = np.stack(outputs)
    self.test_dec_input_weights = np.stack(dec_input_weights)
    print("Load test inputs:            " +str(self.test_inputs.shape))
    print("Load test enc_input_weights: " +str(self.test_enc_input_weights.shape))
    print("Load test outputs:           " +str(self.test_outputs.shape))
    print("Load test dec_input_weights: " +str(self.test_dec_input_weights.shape))

  def read_data(self):
    d = json.loads(open(FLAGS.data_path,'r').read())
    inputs = []
    enc_input_weights = []
    outputs = []
    dec_input_weights = []
    maxlen = 0
    for question in d: 
      questioninputs = []
      questionoutputs = []
      for idx,word in enumerate(question):
        questioninputs.append(word[0])
        if word[2] == 1.0:
          questionoutputs.append(idx+1)
      #inputs.append(questioninputs)
      enc_input_len = len(question) 
      for i in range(FLAGS.max_input_sequence_len-enc_input_len):
        questioninputs.append([0]*801)
      weight = np.zeros(FLAGS.max_input_sequence_len)
      weight[:enc_input_len]=1
      enc_input_weights.append(weight)
      inputs.append(questioninputs)
   
      output=[pointer_net.START_ID]
      for i in questionoutputs:
        # Add 2 to value due to the sepcial tokens
        output.append(int(i)+2)
      output.append(pointer_net.END_ID)
      dec_input_len = len(output)-1
      output += [pointer_net.PAD_ID]*(FLAGS.max_output_sequence_len-dec_input_len)
      output = np.array(output)
      outputs.append(output)
      weight = np.zeros(FLAGS.max_output_sequence_len)
      weight[:dec_input_len]=1
      dec_input_weights.append(weight)
        
    self.inputs = np.stack(inputs)
    self.enc_input_weights = np.stack(enc_input_weights)
    self.outputs = np.stack(outputs)
    self.dec_input_weights = np.stack(dec_input_weights)
    print("Load inputs:            " +str(self.inputs.shape))
    print("Load enc_input_weights: " +str(self.enc_input_weights.shape))
    print("Load outputs:           " +str(self.outputs.shape))
    print("Load dec_input_weights: " +str(self.dec_input_weights.shape))


  def get_batch(self):
    data_size = self.inputs.shape[0]
    sample = np.random.choice(data_size,FLAGS.batch_size,replace=True)
    return self.inputs[sample],self.enc_input_weights[sample],\
      self.outputs[sample], self.dec_input_weights[sample]

  def get_test_batch(self):
    data_size = self.test_inputs.shape[0]
    sample = np.random.choice(data_size,FLAGS.batch_size,replace=True)
    return self.test_inputs[sample],self.test_enc_input_weights[sample],\
      self.test_outputs[sample], self.test_dec_input_weights[sample]

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
      self.writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',self.sess.graph)
      # Try to get checkpoint
      #ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
      #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
      #  print("Load model parameters from %s" % ckpt.model_checkpoint_path)
      #  self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
      #else:
      print("Created model with fresh parameters.")
      self.sess.run(tf.global_variables_initializer())


  def train(self):
    step_time = 0.0
    loss = 0.0
    valloss = 0.0
    current_step = 0
    test_step_loss = 0.0

    while True:
      start_time = time.time()
      inputs,enc_input_weights, outputs, dec_input_weights = \
                  self.get_batch()
      summary, step_loss, predicted_ids_with_logits, targets, debug_var = \
                  self.model.step(self.sess, inputs, enc_input_weights, outputs, dec_input_weights)
      test_inputs,test_enc_input_weights, test_outputs, test_dec_input_weights = \
                  self.get_test_batch()
      test_summary, test_step_loss, test_predicted_ids_with_logits, test_targets, test_debug_var = \
                  self.model.step(self.sess, test_inputs, test_enc_input_weights, test_outputs, test_dec_input_weights)
#      valloss += test_step_loss / FLAGS.steps_per_checkpoint 
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      #DEBUG PART
      #print("debug")
      #print(debug_var)
      #return
      #/DEBUG PART

      #Time to print statistic and save model
      if current_step % FLAGS.steps_per_checkpoint == 0:
        with self.sess.as_default():
          print("inside eval")
          gstep = self.model.global_step.eval()
        print ("global step %d step-time %.2f loss %.2f valloss %.2f" % (gstep, step_time, loss, test_step_loss))
        #print ("global step %d step-time %.2f loss %.2f" % (gstep, step_time, loss))
        #Write summary
        self.writer.add_summary(summary, gstep)
        #Randomly choose one to check
        sample = np.random.choice(FLAGS.batch_size,1)[0]
        print("="*20)
        print("Predict: "+str(np.array(predicted_ids_with_logits[1][sample]).reshape(-1)))
        print("Target : "+str(targets[sample]))
        print("="*20)  
        checkpoint_path = os.path.join(FLAGS.log_dir, "convex_hull.ckpt")
        self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        #self.eval()
        step_time, loss = 0.0, 0.0

  def eval(self):
    """ Randomly get a batch of data and output predictions """  
    print("inside eval again")
    inputs,enc_input_weights, outputs, dec_input_weights = self.get_batch()
    print(inputs.shape, enc_input_weights.shape,self.sess)
    predicted_ids = self.model.step(self.sess, inputs, enc_input_weights)
    print(predicted_ids) 
    print("="*20)
    for i in range(FLAGS.batch_size):
      print("* %dth sample target: %s" % (i,str(outputs[i,1:]-2)))
      for predict in predicted_ids[i]:
        print("prediction: "+str(predict))       
    print("="*20)

  def run(self):
    if self.forward_only:
      self.eval()
    else:
      self.train()

def main(_):
  entitylinker = EntityLinker(FLAGS.forward_only)
  entitylinker.run()

if __name__ == "__main__":
  tf.app.run()
