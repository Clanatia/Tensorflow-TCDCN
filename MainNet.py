#from PrepareData import get_next_batch
import tensorflow as tf
import cv2
import numpy as np
import os

img_size = 250
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=00.1))


def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')

class DataSet(object):
  def __init__(self, images, landmark, gender, smile, glasses, headpose, size, fake_data=False):
    if fake_data:
      self._num_examples = size#13466
    else:
      assert images.shape[0] == landmark.shape[0], ("images.shape: %s landmark.shape: %s" % (images.shape, landmark.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      #assert images.shape[3] == 1
      # Convert from [0, 255] -> [0.0, 1.0].
      #images = images.astype(np.float32)
      #images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._landmark = landmark
    self._gender =gender
    self._smile = smile
    self._glasses = glasses
    self._headpose = headpose
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def landmark(self):
    return self._landmark

  def gender(self):
    return self._gender

  def smile(self):
    return self._smile

  def glasses(self):
    return self._glasses

  def headpose(self):
    return self._headpose

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def get_next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(1600)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._landmark = self._landmark[perm]
      self._gender = self._gender[perm]
      self._smile = self._smile[perm]
      self._glasses = self._glasses[perm]
      self._headpose = self._headpose[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._landmark[start:end], self._gender[start:end], self._smile[start:end], self._glasses[start:end], self._headpose[start:end]


def read_data_sets(fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()
    
  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets

  print("Training read start@")
  train_dir = 'training.txt'  
  VALIDATION_SIZE = 10000
  f = open(train_dir, 'r')
  images_train =np.empty((VALIDATION_SIZE,img_size,img_size,3), dtype=np.float32)
  landmark_train = np.empty((VALIDATION_SIZE,10), dtype=np.float32)
  gender_train = np.empty((VALIDATION_SIZE,2))
  smile_train = np.empty((VALIDATION_SIZE,2))
  glasses_train = np.empty((VALIDATION_SIZE,2))
  headpose_train = np.empty((VALIDATION_SIZE,5))
  
  count = 0
  while True:    
    line = f.readline()
    if not line: break
    str = line.split()
    if not str: break

    img = cv2.imread(str[0])    
    size_x = float(img_size)/float(img.shape[0])
    size_y = float(img_size)/float(img.shape[1])
    if(img.shape[0] != img_size):
      images_train[count] = np.multiply(cv2.resize(img,(img_size, img_size)), 1.0 / 255.0)
    else:
      images_train[count] = img

    landmark_train[count] = [float(str[1])*size_x,float(str[2])*size_y,float(str[3])*size_x,float(str[4])*size_y,float(str[5])*size_x,float(str[6])*size_y,float(str[7])*size_x,float(str[8])*size_y,float(str[9])*size_x,float(str[10])*size_y]
    #if(count ==10): 
    #  print(landmark_train[count])
    gender_train[count][int(str[11])-1] = 1
    smile_train[count][int(str[12])-1] = 1
    glasses_train[count][int(str[13])-1] = 1
    headpose_train[count][int(str[14])-1] = 1
    count+=1
    if(count%100 == 0):
      print(count)
  f.close()

  print("testing read start@")
  train_dir = 'testing.txt'  
  VALIDATION_SIZE = 2995
  f = open(train_dir, 'r')
  images_test =np.empty((VALIDATION_SIZE,img_size,img_size,3), dtype=np.float32)
  landmark_test = np.empty((VALIDATION_SIZE,10), dtype=np.float32)
  gender_test = np.zeros((VALIDATION_SIZE,2))
  smile_test = np.zeros((VALIDATION_SIZE,2))
  glasses_test = np.zeros((VALIDATION_SIZE,2))
  headpose_test = np.zeros((VALIDATION_SIZE,5))
  
  count = 0
  while True:
    line = f.readline()
    if not line: break
    str = line.split()    
    if not str: break    
    
    img = cv2.imread(str[0])        
    size_x = float(img_size)/float(img.shape[0])
    size_y = float(img_size)/float(img.shape[1])
    if(img.shape[0] !=img_size):
      images_test[count] = np.multiply(cv2.resize(img,(img_size, img_size)), 1.0 / 255.0)
    else:
      images_test[count] = img
    landmark_test[count] = [float(str[1])*size_x,float(str[2])*size_y,float(str[3])*size_x,float(str[4])*size_y,float(str[5])*size_x,float(str[6])*size_y,float(str[7])*size_x,float(str[8])*size_y,float(str[9])*size_x,float(str[10])*size_y]
    gender_test[count][int(str[11])-1] = 1
    smile_test[count][int(str[12])-1] = 1
    glasses_test[count][int(str[13])-1] = 1
    headpose_test[count][int(str[14])-1] = 1
    count+=1
  f.close()

  data_sets.train = DataSet(images_train, landmark_train, gender_train, smile_train, glasses_train, headpose_train,count-1)
  data_sets.test = DataSet(images_test, landmark_test, gender_test, smile_test, glasses_test, headpose_test,count-1)
  print(data_sets.train.num_examples, data_sets.test._num_examples)

  return data_sets

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

data = read_data_sets()
print("Read data end!!\n")
image = tf.placeholder(tf.float32, shape=[None, img_size, img_size,3])
landmark = tf.placeholder(tf.float32, shape=[None, 10])
gender = tf.placeholder(tf.float32, shape=[None, 2])
smile = tf.placeholder(tf.float32, shape=[None, 2])
glasses = tf.placeholder(tf.float32, shape=[None, 2])
headpose = tf.placeholder(tf.float32, shape=[None, 5])

# layer 1
W_conv1 = weight_variable([5, 5, 3, 16])
b_conv1 = bias_variable([16])

x_image = tf.reshape(image, [-1, img_size, img_size, 3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) #if 40*40 => -1, 36, 36 16  250=> 246
h_pool1 = max_pool_2x2(h_conv1) # -1 , 18 , 18 , 16 250=> 123

# layer2
W_conv2 = weight_variable([3, 3, 16, 48])
b_conv2 = bias_variable([48])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # -1 , 16 , 16 , 48   123=>121
h_pool2 = max_pool_2x2(h_conv2)  # -1 , 8 , 8 , 48   121 => 60

# layer3
W_conv3 = weight_variable([3, 3, 48, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3) # -1 , 6 , 6 , 64  # 60=> 58
h_pool3 = max_pool_2x2(h_conv3) # -1 , 3 , 3 , 64  58=>29

# layer4
W_conv4 = weight_variable([2, 2, 64, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)  # -1 , 2 , 2 , 64  # 29 => 28
h_pool4 = h_conv4


#  layer5
W_fc1 = weight_variable([int(((((img_size -4)/2 -2)/2 -2)/2 -1)) * int(((((img_size -4)/2 -2)/2 -2)/2 -1)) * 64, 100])
b_fc1 = bias_variable([100])

h_pool4_flat = tf.reshape(h_pool4, [-1, int(((((img_size -4)/2 -2)/2 -2)/2 -1)) * int(((((img_size -4)/2 -2)/2 -2)/2 -1)) * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)  # -1 , 100

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# readout layer

# landmark
W_fc_landmark = weight_variable([100, 10])
b_fc_landmark = bias_variable([10])
y_landmark = tf.matmul(h_fc1_drop, W_fc_landmark) + b_fc_landmark

# gender
W_fc_gender = weight_variable([100, 2])
b_fc_gender = bias_variable([2])
y_gender = tf.matmul(h_fc1_drop, W_fc_gender) + b_fc_gender
# smile
W_fc_smile = weight_variable([100, 2])
b_fc_smile = bias_variable([2])
y_smile = tf.matmul(h_fc1_drop, W_fc_smile) + b_fc_smile
# glasses
W_fc_glasses = weight_variable([100, 2])
b_fc_glasses = bias_variable([2])
y_glasses = tf.matmul(h_fc1_drop, W_fc_glasses) + b_fc_glasses
# headpose
W_fc_headpose = weight_variable([100, 5])
b_fc_headpose = bias_variable([5])
y_headpose = tf.matmul(h_fc1_drop, W_fc_headpose) + b_fc_headpose


error = 1 / 2 * tf.reduce_sum(tf.square(landmark - y_landmark)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_gender   , labels=gender)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_smile    , labels=smile)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_glasses  , labels=glasses)) + \
        tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_headpose , labels=headpose)) + \
        2*tf.nn.l2_loss(W_fc_landmark) + \
        2*tf.nn.l2_loss(W_fc_glasses) + \
        2*tf.nn.l2_loss(W_fc_gender) + \
        2*tf.nn.l2_loss(W_fc_headpose) + \
        2*tf.nn.l2_loss(W_fc_smile)

landmark_error = 1 / 2 * tf.reduce_sum(tf.square(landmark - y_landmark))

# train
train_step = tf.train.AdamOptimizer(1e-3).minimize(error)

saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state("train")

print("Start training\n")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if ckpt and ckpt.model_checkpoint_path:        
        import re
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join("train", ckpt_name))
        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
    else:
        counter = 0
        print(" [*] Failed to find a checkpoint")    

    for x in range(30000):
        i, j, k, l, m, n = data.train.get_next_batch(50)
        error_data = sess.run(error, feed_dict={image: i, landmark: j, gender: k, smile: l, glasses: m, headpose: n, keep_prob: 1})        
        sess.run(        train_step, feed_dict={image: i, landmark: j, gender: k, smile: l, glasses: m, headpose: n, keep_prob: 0.9})

        if(x%100 ==0) :
          print("idle:",x+counter, ",err :",error_data)
          o, p, q, r, s, t = data.test.get_next_batch(50)
          #print(j[0],k[0],l[0],m[0],n[0])
          #print(p[0],q[0],r[0],s[0],t[0])
          Landmarkerror=sess.run(landmark_error,feed_dict={image: o, landmark: p, gender: q, smile: r, glasses: s, headpose: t, keep_prob: 1})
          HoleError = sess.run(error        , feed_dict={image: o, landmark: p, gender: q, smile: r, glasses: s, headpose: t, keep_prob: 1})         
          print("landmark error :",Landmarkerror)
          print("Hole error :",HoleError)
    
    model_name = "model"
    checkpoint_dir = "train"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print("total training count :",counter+x)
    saver.save(sess,os.path.join(checkpoint_dir, model_name), global_step=counter+x)

    print(sess.run(y_landmark        , feed_dict={image: o, keep_prob: 1})[0])
    print(p[0])
    print(softmax(sess.run(y_gender        , feed_dict={image: o, keep_prob: 1})[0]))
    print(q[0])
    print(softmax(sess.run(y_smile        , feed_dict={image: o, keep_prob: 1})[0]))
    print(r[0])
    print(softmax(sess.run(y_glasses        , feed_dict={image: o, keep_prob: 1})[0]))
    print(s[0])
    print(softmax(sess.run(y_headpose        , feed_dict={image: o, keep_prob: 1})[0]))
    print(t[0])