import tensorflow as tf
import numpy as np
import pointer_net 
import time
import os
import sys
import json

tf.app.flags.DEFINE_integer("batch_size", 20,"Batch size.")
tf.app.flags.DEFINE_integer("max_input_sequence_len", 1000, "Maximum input sequence length.")
tf.app.flags.DEFINE_integer("max_output_sequence_len", 100, "Maximum output sequence length.")
tf.app.flags.DEFINE_integer("rnn_size", 128, "RNN unit size.")
tf.app.flags.DEFINE_integer("attention_size", 500, "Attention size.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers.")
tf.app.flags.DEFINE_integer("beam_width", 1, "Width of beam search .")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Maximum gradient norm.")
tf.app.flags.DEFINE_boolean("forward_only", False, "Forward Only.")
#tf.app.flags.DEFINE_string("log_dir", "./log", "Log directory")
tf.app.flags.DEFINE_string("data_path", "./earl2datasets/pctrainlcquad2.txt", "Training Data path.")
tf.app.flags.DEFINE_string("test_data_path", "./earl2datasets/pctestlcquad2.txt", "Test Data path.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 5, "frequence to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS
FLAGS.log_dir = "./lcquad2logs4"

class EntityLinker(object):
  def __init__(self, forward_only):
    self.forward_only = forward_only
    self.graph = tf.Graph()
    with self.graph.as_default():
      config = tf.ConfigProto()
      config.gpu_options.allow_growth=True
      self.sess = tf.Session(config=config)
    self.read_test_data()
    self.build_model()
    

  def read_test_data(self): 
    inputs = []
    enc_input_weights = []
    outputs = []
    dec_input_weights = []
    maxlen = 0
    with open(FLAGS.test_data_path,'r') as fp:
      for line in fp:
        line = line.strip()
        question = json.loads(line)
        questioninputs = []
        questionoutputs = []
        for idx,word in enumerate(question):
          questioninputs.append(word[0])
          if word[2] == 1.0:
            questionoutputs.append(idx+1)
      #inputs.append(questioninputs)
        enc_input_len = len(question)
        if enc_input_len > FLAGS.max_input_sequence_len:
          continue
        for i in range(FLAGS.max_input_sequence_len-enc_input_len):
          questioninputs.append([0]*802)
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
        if len(inputs) == FLAGS.batch_size:
          break

    self.test_inputs = np.stack(inputs)
    self.test_enc_input_weights = np.stack(enc_input_weights)
    self.test_outputs = np.stack(outputs)
    self.test_dec_input_weights = np.stack(dec_input_weights)
#    print("Load test inputs:            " +str(self.test_inputs.shape))
#    print("Load test enc_input_weights: " +str(self.test_enc_input_weights.shape))
#    print("Load test outputs:           " +str(self.test_outputs.shape))
#    print("Load test dec_input_weights: " +str(self.test_dec_input_weights.shape))


  def read_data(self, step):
    inputs = []
    enc_input_weights = []
    outputs = []
    dec_input_weights = []
    maxlen = 0
    linecount = 0
    with open(FLAGS.data_path,'r') as fp:
      for line in fp:
        linecount += 1
        if linecount < step*FLAGS.batch_size:
          continue
        line = line.strip()
        question = json.loads(line)
        questioninputs = []
        questionoutputs = []
        for idx,word in enumerate(question):
          questioninputs.append(word[0])
          if word[2] == 1.0:
            questionoutputs.append(idx+1)
      #inputs.append(questioninputs)
        enc_input_len = len(question) 
        if enc_input_len > FLAGS.max_input_sequence_len:
          continue
        for i in range(FLAGS.max_input_sequence_len-enc_input_len):
          questioninputs.append([0]*802)
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
        if len(inputs) >= FLAGS.batch_size:
          break
        
    self.inputs = np.stack(inputs)
    self.enc_input_weights = np.stack(enc_input_weights)
    self.outputs = np.stack(outputs)
    self.dec_input_weights = np.stack(dec_input_weights)
#    print("Load inputs:            " +str(self.inputs.shape))
#    print("Load enc_input_weights: " +str(self.enc_input_weights.shape))
#    print("Load outputs:           " +str(self.outputs.shape))
#    print("Load dec_input_weights: " +str(self.dec_input_weights.shape))


  def get_batch(self, step):
    self.read_data(step)
    return self.inputs,self.enc_input_weights,\
      self.outputs, self.dec_input_weights

  def get_test_batch(self):
    return self.test_inputs,self.test_enc_input_weights,\
      self.test_outputs, self.test_dec_input_weights

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
      self.writer = tf.summary.FileWriter(FLAGS.log_dir + '/train',self.sess.graph)
      print("Created model with fresh parameters.")
      self.sess.run(tf.global_variables_initializer())


  def train(self):
    step_time = 0.0
    loss = 0.0
    valloss = 0.0
    current_step = 0
    test_step_loss = 0.0
    besttestloss = 99999
    while True:
      start_time = time.time()
      inputs,enc_input_weights, outputs, dec_input_weights = \
                  self.get_batch(current_step)
      summary, step_loss, predicted_ids_with_logits, targets, debug_var = \
              self.model.step(self.sess, inputs, enc_input_weights, outputs, dec_input_weights, update=True)
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      #Time to print statistic and save model
      if current_step % FLAGS.steps_per_checkpoint == 0:
        test_inputs,test_enc_input_weights, test_outputs, test_dec_input_weights = \
                  self.get_test_batch()
        test_summary, test_step_loss, test_predicted_ids_with_logits, test_targets, test_debug_var = \
                  self.model.step(self.sess, test_inputs, test_enc_input_weights, test_outputs, test_dec_input_weights, update=False)
        with self.sess.as_default():
          gstep = self.model.global_step.eval()
        print ("global step %d step-time %.2f loss %.2f valloss %.2f" % (gstep, step_time, loss, test_step_loss))
        #print ("global step %d step-time %.2f loss %.2f" % (gstep, step_time, loss))
        #Write summary
        self.writer.add_summary(summary, gstep)
        #Randomly choose one to check
        sample = np.random.choice(FLAGS.batch_size,1)[0]
        print("="*20)
        print("Predict: "+str(np.array(test_predicted_ids_with_logits[1][sample]).reshape(-1)))
        print("Target : "+str(test_targets[sample]))
        print("="*20)  
        checkpoint_path = os.path.join(FLAGS.log_dir, "convex_hull.ckpt")
        if current_step % FLAGS.steps_per_checkpoint*10 == 0:
          self.model.saver.save(self.sess, checkpoint_path, global_step=self.model.global_step)
        #self.eval()
        step_time, loss = 0.0, 0.0

  def run(self):
    self.train()

def main(_):
  entitylinker = EntityLinker(FLAGS.forward_only)
  entitylinker.run()

if __name__ == "__main__":
  tf.app.run()
