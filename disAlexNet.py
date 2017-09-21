#-*-coding:UTF-8-*-
from __future__ import print_function

import tensorflow as tf
import sys
import time
import os
from AlexNet import AlexNet
import train_util as tu
#get the optimizer
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['GRPC_VERBOSITY_LEVEL']='DEBUG'

def get_optimizer(optimizer, learning_rate):
    if optimizer == "SGD":
	return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == "Adadelta":
	return  tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == "Adagrad":
	return  tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == "Ftrl":
        return  tf.train.FtrlOptimizer(learning_rate)
    elif optimizer == "Adam":
	return  tf.train.AdamOptimizer(learning_rate)
    elif optimizer == "Momentum":
	return  tf.train.MomentumOptimizer(learning_rate)
    elif optimizer == "RMSProp":
	return  tf.train.RMSProp(learning_rate)

# cluster specification
parameter_servers = sys.argv[1].split(',')
n_PS = len(parameter_servers)
workers = sys.argv[2].split(',')
n_Workers = len(workers)
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_float("targted_accuracy", 0.5, "targted accuracy of model")
tf.app.flags.DEFINE_string("optimizer", "SGD", "optimizer we adopted")
tf.app.flags.DEFINE_integer("Batch_size", 128, "Batch size")
tf.app.flags.DEFINE_float("Learning_rate", 0.0001, "Learning rate")
tf.app.flags.DEFINE_integer("Epoch", 10, "Epoch")
tf.app.flags.DEFINE_string("imagenet_path", 10, "ImageNet data path")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index)
	
# config
batch_size = FLAGS.Batch_size
learning_rate = FLAGS.Learning_rate
targted_accuracy = FLAGS.targted_accuracy
Optimizer = FLAGS.optimizer
Epoch = FLAGS.Epoch
imagenet_path = FLAGS.imagenet_path

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
	#More to come on is_chief...
        is_chief = FLAGS.task_index == 0
	# count the number of global steps
	global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)

	# load ImageNet-1k data set
	#train_img_path = os.path.join(imagenet_path, 'ILSVRC2012_img_train')
	#ts_size = tu.imagenet_size(train_img_path)
	num_batches = int(float(5000) / batch_size)
	#wnid_labels, _ = tu.load_imagenet_meta(os.path.join(imagenet_path, 'data/meta.mat'))
	#-----------------------------------TUDO Check data input-------------------------------------------------#
	
	# input images
	with tf.name_scope('input'):
	    # None -> batch size can be any size, [224, 224, 3] -> image
	    x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="x-input")
	    # target 10 output classes
	    y_ = tf.placeholder(tf.float32, shape=[None, 1000], name="y-input")
	
	#creat an AlexNet
  	y_conv, _ = alexnet.classifier(x_b, keep_prob)

	# specify cost function
	with tf.name_scope('cross_entropy'):
	    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	
	# specify optimizer
	with tf.name_scope('train'):
	    grad_op = get_optimizer( Optimizer, learning_rate)
	    train_op = grad_op.minimize(cross_entropy, global_step=global_step)
	
	# accuracy
	with tf.name_scope('Accuracy'):
	    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	init_op = tf.global_variables_initializer()
	variables_check_op=tf.report_uninitialized_variables()

    	sess_config = tf.ConfigProto(
        	allow_soft_placement=True,
        	log_device_placement=False,
        	device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])
    sv = tf.train.Supervisor(is_chief=is_chief,
			     init_op=init_op,
                             global_step=global_step)
    server_grpc_url = "grpc://" + workers[FLAGS.task_index]
    state = False
    with sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config) as sess:
	while(not state):
            uninitalized_variables=sess.run(variables_check_op)
	    if(len(uninitalized_variables.shape) == 1):
		state = True
	
	step = 0
	cost = 0
	final_accuracy = 0
	start_time = time.time()
	batch_time = time.time()
	epoch_time = time.time()
	while (not sv.should_stop()):
	    #Read batch_size data
	    for e in range(epochs):
		for i in range(num_batches):
		    #batch_x, batch_y = tu.read_batch(batch_size, train_img_path, wnid_labels)
                    batch_x, batch_y = tu.read_validation_batch(batch_size, os.path.join(imagenet_path, 'ILSVRC2012_img_val'), os.path.join(imagenet_path, '/ILSVRC2012_validation_ground_truth.txt'))
                    _, cost, step = sess.run([train_op, cross_entropy, global_step], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
	    	    print("Step: %d," % (step+1), 
			        " Accuracy: %.4f," % final_accuracy,
			        " Loss: %f" % cost,
			        " Bctch_Time: %fs" % float(time.time()-begin_time))
	    	    batch_time = time.time()

	    	print("Epoch: %d," % (e+1), 
			" Accuracy: %.4f," % final_accuracy,
			" Loss: %f" % cost,
			" Epoch_Time: %fs" % float(time.time()-epoch_time),
			" Tolal_Time: %fs" % float(time.time()-start_time))
		
	#index, sum_step, total_time, cost, final_accuracy    
	#final_accuracy = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
	#re = str(n_PS) + '-' + str(n_Workers) + '-' + str(FLAGS.task_index) + ',' + str(step) + ',' + str(float(time.time()-start_time)) + ',' + str(cost) + ',' + str(final_accuracy)
       # writer = open("re_2_"+Optimizer+".csv", "a+")
        #writer.write(re+"\r\n")
       # writer.close()
    sv.stop 
