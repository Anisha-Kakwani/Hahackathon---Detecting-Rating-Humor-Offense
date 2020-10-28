#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install tensorflow


# In[ ]:


#pip install tensorflow_hub


# In[ ]:


import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import pandas as pd

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[ ]:


#ls


# In[ ]:


dataset = pd.read_csv('Kaggle-dataset.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:,1].values


# In[ ]:


label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)


# In[ ]:


train_examples, test_examples, train_labels, test_labels = train_test_split(X,Y, test_size=0.1,random_state=0)
print("Training entries: {}, test entries: {}".format(len(train_examples), len(test_examples)))
print("TRAIN Dataset -> ")
print(train_examples[:10])
print(" Test Dataset ->")
print(train_labels[:10])


# # <font color="green">Build the model </font> <br>
# The neural network is created by stacking layers—this requires three main architectural decisions:
#  * How to represent the text?
#  * How many layers to use in the model?
#  * How many *hidden units* to use for each layer? 
# In this example, the input data consists of sentences. The labels to predict are either 0 or 1.
# 
# One way to represent the text is to convert sentences into embeddings vectors. We can use a pre-trained text embedding as the first layer, which will have two advantages:
#  *   we don't have to worry about text preprocessing,
#  *   we can benefit from transfer learning.
#  
# For this example we will use a model from [TensorFlow Hub] (https://www.tensorflow.org/hub) called [google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1).
# There are three other models to test for the sake of this tutorial:
#  * [google/tf2-preview/gnews-swivel-20dim-with-oov/1] (https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1) - same as [google/tf2-preview/gnews-swivel-20dim/1] (https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1), but with 2.5% vocabulary converted to OOV buckets. This can help if vocabulary of the task and vocabulary of the model don't fully overlap.
#  * [google/tf2-preview/nnlm-en-dim50/1] (https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1) - A much larger model with ~1M vocabulary size and 50 dimensions.
#  * [google/tf2-preview/nnlm-en-dim128/1] (https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1) - Even larger model with ~1M vocabulary size and 128 dimensions.
# 
# Let's first create a Keras layer that uses a TensorFlow Hub model to embed the sentences, and try it out on a couple of input examples. Note that the output shape of the produced embeddings is a expected: `(num_examples, embedding_dimension)`.
# 

# In[ ]:


model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(model, output_shape=[20], input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples[:3])


# <font color="green">Let's now build the full model: </font>

# In[ ]:


model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.summary()


# <font color="green"> The layers are stacked sequentially to build the classifier: </font> <br>
#  
#  1. The first layer is a TensorFlow Hub layer. This layer uses a pre-trained Saved Model to map a sentence into its embedding vector. The model that we are using ([google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1)) splits the sentence into tokens, embeds each token and then combines the embedding. The resulting dimensions are: `(num_examples, embedding_dimension)`.
#  2. This fixed-length output vector is piped through a fully-connected (`Dense`) layer with 16 hidden units.
#  3. The last layer is densely connected with a single output node. This outputs logits: the log-odds of the true class, according to the model.
# 
# <font color="green"> Hidden units </font> <br>
#  
# The above model has two intermediate or "hidden" layers, between the input and output. 
# The number of outputs (units, nodes, or neurons) is the dimension of the representational space for the layer. In other words, the amount of freedom the network is allowed when learning an internal representation.
#  
# If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to learning unwanted patterns—patterns that improve performance on training data but not on the test data. This is called *overfitting*, and we'll explore it later.
# 
# <font color="green">Loss function and optimizer </font> <br>
#  
# A model needs a loss function and an optimizer for training. Since this is a binary classification problem and the model outputs a probability (a single-unit layer with a sigmoid activation), we'll use the `binary_crossentropy` loss function. 
#  
# This isn't the only choice for a loss function, you could, for instance, choose `mean_squared_error`. But, generally, `binary_crossentropy` is better for dealing with probabilities—it measures the "distance" between probability distributions, or in our case, between the ground-truth distribution and the predictions.
#  
# Later, when we are exploring regression problems (say, to predict the price of a house), we will see how to use another loss function called mean squared error.
#  
# <font color="green">Now, configure the model to use an optimizer and a loss function: </font>

# In[ ]:


model.compile(optimizer='adam',
              loss=tf.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])


# <font color="green">Create a validation set </font>
# 
# When training, we want to check the accuracy of the model on data it hasn't seen before. <br>
# Create a *validation set* by setting apart 10,000 examples from the original training data. <br> 
# (Why not use the testing set now? Our goal is to develop and tune our model using only the training data, then use the test data just once to evaluate our accuracy).

# In[ ]:


x_val = train_examples[:20000]
partial_x_train = train_examples[20000:]

y_val = train_labels[:20000]
partial_y_train = train_labels[20000:]


# <font color="green">Train the model </font> <br>
# Train the model for 40 epochs in mini-batches of 512 samples. This is 40 iterations over all samples in the `x_train` and `y_train` tensors. While training, monitor the model's loss and accuracy on the 10,000 samples from the validation set:

# In[ ]:


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# <font color="green"> Evaluate the model </font> <br>
# And let's see how the model performs. Two values will be returned. <br>
# -> Loss (a number which represents our error, lower values are better), and <br>
# -> accuracy.

# In[ ]:


results = model.evaluate(test_examples, test_labels)

print("*******KAGGLE************",results)


# In[ ]:


codalab_data = pd.read_csv('Codalab-trial-dataset.csv')
test_data_codalab = codalab_data.iloc[:, 1].values
test_labels_codalab = codalab_data.iloc[:,2].values

results_codalab = model.evaluate(test_data_codalab, test_labels_codalab)

print("*******CODALAB************",results_codalab) 


# # need to change
# This fairly naive approach achieves an accuracy of about 87%. With more advanced approaches, the model should get closer to 95%.<br>
# 
# Create a graph of accuracy and loss over time<br>
# `model.fit()` returns a `History` object that contains a dictionary with everything that happened during training:

# In[ ]:


history_dict = history.history
history_dict.keys()


# # need to change
# There are four entries: one for each monitored metric during training and validation. We can use these to plot the training and validation loss for comparison, as well as the training and validation accuracy:

# In[ ]:


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)


# In[ ]:


# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# In[ ]:


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# 
# In this plot, the dots represent the training loss and accuracy, and the solid lines are the validation loss and accuracy.
# 
# Notice the training loss *decreases* with each epoch and the training accuracy *increases* with each epoch. This is expected when using a gradient descent optimization—it should minimize the desired quantity on every iteration.
# 
# This isn't the case for the validation loss and accuracy—they seem to peak after about twenty epochs. This is an example of overfitting: the model performs better on the training data than it does on data it has never seen before. After this point, the model over-optimizes and learns representations *specific* to the training data that do not *generalize* to test data.
# 
# For this particular case, we could prevent overfitting by simply stopping the training after twenty or so epochs. Later, you'll see how to do this automatically with a callback.

# In[28]:


codalab_data.columns


# In[67]:


import pandas as pd

codalab_data = pd.read_csv('Codalab-trial-dataset.csv')
print(codalab_data.head())


# In[68]:


from sklearn.impute import SimpleImputer 
constant_imputer=SimpleImputer(strategy='median')
codalab_data.iloc[:,2:24]=constant_imputer.fit_transform(codalab_data.iloc[:,2:24])
#x_array = np.array(codalab_data['funniness_all'])
#normalized_X = preprocessing.normalize([x_array])
print(codalab_data.isnull().sum())


# In[21]:


from sklearn.impute import SimpleImputer 
constant_imputer=SimpleImputer(strategy='most_frequent')
codalab_data.iloc[:,:]=constant_imputer.fit_transform(codalab_data)
#x_array = np.array(codalab_data['funniness_all'])
#normalized_X = preprocessing.normalize([x_array])
print(codalab_data.isnull().sum())


# In[29]:


from sklearn.impute import SimpleImputer 
constant_imputer=SimpleImputer(strategy='constant', fill_value=0)
codalab_data.iloc[:,:]=constant_imputer.fit_transform(codalab_data)
#x_array = np.array(codalab_data['funniness_all'])
#normalized_X = preprocessing.normalize([x_array])
print(codalab_data.isnull().sum())


# In[72]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

X = codalab_data.iloc[:, 3:23].values
Y = codalab_data.iloc[:,2].values
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

kfold = KFold(n_splits=5, random_state=110,shuffle=True)
model = LogisticRegression()
results = cross_val_score(model, X, Y, cv=kfold,scoring='accuracy')
print("Accuracy in each split:",results)
print("Accuracy:", results.mean())
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))







# In[77]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

X = codalab_data.iloc[:, 3:23].values
Y = codalab_data.iloc[:,2].values
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

kf=KFold(n_splits=10,random_state=1050,shuffle=True)
splits=kf.split(X)

rfr = RandomForestRegressor(n_estimators=25)
scores=[]
for train_index, test_index in splits:
    X_train, y_train = X[train_index], Y[train_index]
    X_test, y_test = X[test_index], Y[test_index]

    rfr.fit(X_train, y_train)
    predictions = rfr.predict(X_test)
    print("Mean squared error: " + str(mean_squared_error(y_test, predictions)))
    scores.append((rfr.score(X_test, y_test)))
print("Accuracy:",np.mean(scores))
    


# In[ ]:


from sklearn.model_selection import ShuffleSplit
from sklearn import svm
n_samples = X.shape[0]
clf = svm.SVC(kernel='linear', C=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

