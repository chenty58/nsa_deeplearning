import tensorflow as tf
import numpy as np
import pandas as pd
from numpy import genfromtxt
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
import pandas as pd

###############################################################

learning_rate = 0.01
n_epochs = 5000
batch_size = 100000

def convertOneHot(data,num):
    tensor = tf.one_hot(data,num)
    ss = tf.Session()
    array = ss.run(tensor)
    return array

###############################################################
print("Begin:__________________________________")
featuer_col = np.arange(41)
feature = pd.read_csv('kddcup.data.csv',delimiter=',',usecols=(featuer_col),dtype=str).values
for c in range(1,4):
    feature[:,c]=LabelEncoder().fit_transform(feature[:,c])
feature=feature.astype(np.float)
target=pd.read_csv('kddcup.data.csv',delimiter=',',usecols=([41]),dtype=str).values
for i in range(0,len(target)):
    if target[i] != 'normal.':
        target[i] = 'not_normal'
sc = StandardScaler()
sc.fit(feature)
feature_std = sc.transform(feature)
target_label = LabelEncoder().fit_transform(target)
target_onehot = convertOneHot(target_label,2)
x_train, x_test, y_train_onehot, y_test_onehot = train_test_split(feature_std, target_onehot, test_size=0.10, random_state=0)
A=x_train.shape[1]
B=len(y_train_onehot[0])
###################################################
## print stats 
precision_scores_list = []
accuracy_scores_list = []

def print_stats_metrics(y_test, y_pred):    
    print('Accuracy: %.2f' % accuracy_score(y_test,   y_pred) )
    #Accuracy: 0.84
    accuracy_scores_list.append(accuracy_score(y_test,   y_pred) )
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print ("confusion matrix")
    print(confmat)
    print (pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    precision_scores_list.append(precision_score(y_true=y_test, y_pred=y_pred))
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1-measure: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))

###############################################################


def layer(input, weight_shape, bias_shape):
    weight_stddev = (2.0/weight_shape[0])**0.5
    w_init = tf.random_normal_initializer(stddev=weight_stddev)
    bias_init = tf.constant_initializer(value=0)
    W = tf.get_variable("W", weight_shape, initializer=w_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    return tf.nn.relu(tf.matmul(input, W) + b)
###############################################################
def inference_deep_layers(x_tf, A, B):
    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x_tf, [A, 30],[30])
    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, [30, 20],[20])
    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, [20, 15],[15])
    with tf.variable_scope("hidden_4"):
        hidden_4 = layer(hidden_3, [15, 10],[10])
    with tf.variable_scope("output"):
        output = layer(hidden_4, [10, B], [B])
    return output
###############################################################
def loss_deep(output, y_tf):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_tf)
    loss = tf.reduce_mean(xentropy) 
    return loss
###########################################################

def training(cost):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(cost)
    return train_op

###########################################################
def evaluate(output, y_tf):
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_tf,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy
###############################################################

x_tf = tf.placeholder("float",[None,A])
y_tf = tf.placeholder("float",[None,B])
###############################################################
output = inference_deep_layers(x_tf,A,B)
cost = loss_deep(output,y_tf)
train_op=training(cost)
eval_op=evaluate(output,y_tf)
###############################################################
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
###############################################################
y_p_metrics = tf.argmax(output,1)
###############################################################
num_samples_train_set=x_train.shape[0]
num_batches = int(num_samples_train_set/batch_size)

###############################################################

for i in range(n_epochs):
    print("epoch %s out of %s"%(i,n_epochs))
    for batch_n in range(num_batches):
        sta = batch_n*batch_size
        end = sta+batch_size
        sess.run(train_op,feed_dict={x_tf:x_train[sta:end],y_tf:y_train_onehot[sta:end]})
    print ("-------------------------------------------------------------------------------")    
    print ("Accuracy score")
    result, y_result_metrics = sess.run([eval_op, y_p_metrics], feed_dict={x_tf: x_test, y_tf: y_test_onehot})
    print("Run {},{}".format(i,result))
    y_true = np.argmax(y_test_onehot,1)
    print_stats_metrics(y_true, y_result_metrics)


