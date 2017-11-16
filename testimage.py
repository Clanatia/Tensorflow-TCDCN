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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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

print("Start test\n")
img = cv2.imread('lfw_5590\Aaron_Eckhart_0001.jpg')    
size_x = float(img_size)/float(img.shape[0])
size_y = float(img_size)/float(img.shape[1])
print(size_x,size_y)

test_image =np.empty((1,img_size,img_size,3), dtype=np.float32)
if(img.shape[0] != img_size):
    test_image[0] = np.multiply(cv2.resize(img,(img_size, img_size)), 1.0 / 255.0)
else: 
    test_image[0] = img

landmark_test = np.empty((10), dtype=np.float32)
gender_test = np.zeros((2))
smile_test = np.zeros((2))
glasses_test = np.zeros((2))
headpose_test = np.zeros((5))

train_dir = 'training.txt'      
f = open(train_dir, 'r')
line = f.readline()
str = line.split()
landmark_test = [float(str[1])*size_x,float(str[2])*size_y,float(str[3])*size_x,float(str[4])*size_y,float(str[5])*size_x,float(str[6])*size_y,float(str[7])*size_x,float(str[8])*size_y,float(str[9])*size_x,float(str[10])*size_y]
gender_test[int(str[11])-1] = 1
smile_test[int(str[12])-1] = 1
glasses_test[int(str[13])-1] = 1
headpose_test[int(str[14])-1] = 1

#cap = cv2.VideoCapture(0)  #parameta 0은 컴퓨터와 연결된 usb cam설정을 불러옴
#ret, frame = cap.read() 
#size_x = float(img_size)/float(frame.shape[0])
#size_y = float(img_size)/float(frame.shape[1])

colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

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
       
    #while(True):
    #    ret, frame = cap.read()
    #    if(frame.shape[0] != img_size):
    #        test_image[0] = np.multiply(cv2.resize(frame,(img_size, img_size)), 1.0 / 255.0)
    #    else:
    #        test_image[0] = frame
    #    landmarks = sess.run(y_landmark        , feed_dict={image: test_image, keep_prob: 1})[0]
        
    #    for i in range(5):
    #        cv2.circle(frame, (landmarks[i*2],landmarks[i*2+1]) , 4, colors[i], thickness=-1)

    #    cv2.imshow("view", frame)
    #    if cv2.waitKey(30) & 0xFF == ord('q'): #waitKey의 value의 값이 너무 높으면 너무 자주 key값 확인을 해서 느려짐, 프레임에 맞게 설정
    #        break        
        
    #cap.release()
    #cv2.destroyAllWindows()
    
    landmarks = sess.run(y_landmark, feed_dict={image: test_image, keep_prob: 1})[0]
        
    for i in range(5):
        print(int(landmarks[i]/size_x),int(landmarks[i+5]/size_y))
        print(int(landmark_test[i]/size_x),int(landmark_test[i+5]/size_y))
        cv2.circle(img, ( int(landmarks[i]/size_x),int(landmarks[i+5]/size_y)) , 4, colors[i], thickness=-1)
    
    cv2.imshow("view", img)
    cv2.waitKey()
    
    #print(softmax(sess.run(y_gender  , feed_dict={image: test_image, keep_prob: 1})[0]))
    #print(gender_test)
    #print(softmax(sess.run(y_smile , feed_dict={image: test_image, keep_prob: 1})[0]))
    #print(smile_test)
    #print(softmax(sess.run(y_glasses , feed_dict={image: test_image, keep_prob: 1})[0]))
    #print(glasses_test)
    #print(softmax(sess.run(y_headpose , feed_dict={image: test_image, keep_prob: 1})[0]))
    #print(headpose_test)