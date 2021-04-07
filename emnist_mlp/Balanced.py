import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
from keras import layers
from keras import Sequential
import random
from sklearn.model_selection import train_test_split as Split
from sklearn.metrics import confusion_matrix, f1_score

###############################################################################
#Import train data for balanced
x_train_bal = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-balanced-train-images-idx3-ubyte')
y_train_bal = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-balanced-train-labels-idx1-ubyte') 
label_train_bal = np.loadtxt('EMNIST_Raw_Data/emnist-balanced-mapping.txt' ,delimiter=" ")
#Import test data for balanced
x_test_bal = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-balanced-test-images-idx3-ubyte')
y_test_bal = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-balanced-test-labels-idx1-ubyte')

print(x_train_bal.shape, y_train_bal.shape)
print(x_test_bal.shape, y_test_bal.shape)

#Normalise images to [1,0]
x_train_bal = x_train_bal/255.0 
x_test_bal = x_test_bal/255.0

#Transpose Images
for i in range(112800):
    x_train_bal[i]=np.transpose(x_train_bal[i])

for i in range(18800):
    x_test_bal[i]=np.transpose(x_test_bal[i])

###############################################################################
#Build Model
def modelbuilding_function(layers_no, input_nodes, output_nodes, activ_func, 
                           activ_out_func, optim = "adam", 
                           lossfunc='sparse_categorical_crossentropy', 
                           met=['accuracy']):

    model = Sequential()
    model.add(layers.Flatten(input_shape=(28,28)))
    model.add(layers.Dropout(0.2))
    for i in range(layers_no):
        model.add(layers.Dense(input_nodes, activation=activ_func))
    model.add(layers.Dense(output_nodes, activation = activ_out_func))
    model.compile(optimizer=optim,
                  loss=lossfunc,
                  metrics=met)
    return model
###############################################################################
#Determine best model parameters
indep_bal = x_train_bal
depen_bal = y_train_bal

x_training, x_val, y_training, y_val = Split(indep_bal, depen_bal, test_size = 0.1)

#Run a combination of epochs and batchsizes to determine best accuracy.
epochs = [10,20]
batches = [128,256,512]

print("This might take time to run!")
score = []
bestScore = 0
for i in batches:
    for j in epochs:
        random.seed(123)
        model_bal = modelbuilding_function(layers_no=3, input_nodes=256,
                                       output_nodes = 47, activ_func='relu',
                                       activ_out_func='softmax',
                                       lossfunc='sparse_categorical_crossentropy',
                                       met=['accuracy'])
        model_bal.fit(x_training, y_training, epochs = j, batch_size= i, verbose=0)
        scores = model_bal.evaluate(x_test_bal, y_test_bal, verbose=0)
        score.append(scores[1]*100);
        
        if scores[1]>bestScore:
            bestScore=scores[1]
            print("%.2f%% %10i %10i" % (bestScore*100,i,j))
print("\nall done")            
###############################################################################
#Fit best model and test accuracy
model_bal.fit(x_train_bal, y_train_bal, epochs=20, batch_size = 256)
acc_bal = model_bal.evaluate(x_test_bal, y_test_bal, verbose=0)
print("Batch Size = 256, Epochs = 20 for best model accuracy")
print("Balanced Model Accuracy: %.2f%%"%(acc_bal[1]*100))
print("Model Summary")
model_bal.summary()
###############################################################################
#Print some predictions of test data using training model
x_pred_bal = np.argmax(model_bal.predict(x_test_bal),axis=1)
for i in range(5):
    print("\nThe following digit is a ",x_pred_bal[i])
    plt.imshow(x_test_bal[i])
    plt.show()
###############################################################################
#Validation Data
x_val_bal = x_train_bal[:18800]
partial_x_train_bal = x_train_bal[18800:]
y_val_bal = y_train_bal[:18800] 
partial_y_train_bal = y_train_bal[18800:]

#Find St.dev and avg. accuracy accross models
for i in range(10):  
    x_train_bal,x_val_bal,y_train_bal,y_val_bal = Split(partial_x_train_bal,
                                                        partial_y_train_bal,test_size=0.9)    
    model_bal.fit(x_train_bal, y_train_bal, epochs=20, batch_size=256,verbose=0)
    scores = model_bal.evaluate(x_val_bal, y_val_bal,verbose=0)
    score.append(scores[1]*100);
    
print("Average accuracy across independantly trained models : %.2f%%"%np.average(score))
print("Std Dev accuracy across independantly trained models : %.2f%%"%np.std(score))

plt.plot(score)
plt.title('Accuracy across all models')
plt.grid()
plt.show

#Validation results
history = model_bal.fit(partial_x_train_bal,partial_y_train_bal,epochs=20,
                        batch_size=256,validation_data=(x_val_bal, y_val_bal))

#Save the training history
hist_dict_bal = history.history
acc_bal = history.history['acc']
val_acc_bal = history.history['val_acc']
loss_values_bal = hist_dict_bal['loss']
val_loss_values_bal = hist_dict_bal['val_loss']
epochs_bal = range(1, len(acc_bal) + 1)

#Determine max accuracy of validation test
max_bal = max(val_acc_bal)
print(max_bal)
epoch_acc_bal = val_acc_bal.index(max_bal)
print(epoch_acc_bal)
dev_val_bal=np.std(val_acc_bal)

print("Best Validation Accuracy", max_bal, "at Epoch", epoch_acc_bal)
###############################################################################
# Plot the training and validation losses
plt.plot(epochs_bal, loss_values_bal, 'bo', label='Training loss')
plt.plot(epochs_bal, val_loss_values_bal, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(ymin=0, ymax=0.8)
plt.legend()
plt.show()

#Plot the training and validation accuracies

plt.clf()
acc_values = hist_dict_bal['acc']
val_acc_values = hist_dict_bal['val_acc']
plt.plot(epochs_bal, acc_bal, 'bo', label='Training acc')
plt.plot(epochs_bal, val_acc_bal, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim(ymin=0,ymax=1)
plt.show()
###############################################################################
#Confusion Matrix
import pandas as pd
y_pred_bal= model_bal.predict(x_test_bal)
y_pred_bal = np.argmax(y_pred_bal, axis = 1)
cm_bal = confusion_matrix(y_test_bal, y_pred_bal)
print("Confusion matrix:\n%s" % cm_bal)
my_df = pd.DataFrame(cm_bal) 
my_df.to_csv('matbal.csv', index=False)

f1_score(y_test_bal, y_pred_bal, average='micro')

###############################################################################

def analysis(classes):
    














    