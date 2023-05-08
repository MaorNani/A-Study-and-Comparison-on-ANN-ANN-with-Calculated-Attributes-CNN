'''Import Libraries'''
#import relevent libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import numpy as np
import math as m
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report
import time


'''Program functions'''
#Function that creates visualization for existing images in the database
#according to the training / test values of the user's choice
#(according to a defined amount of images)
def sample_plot (x, y, classes, start_index=0, end_index=0):
    index_list=list(range(start_index, end_index+1))
    fig=plt.figure(figsize=(10, 6))
    cols = m.ceil(m.sqrt(len(index_list)))
    rows = m.ceil(len(index_list) / cols)
    pos=0
    for index in index_list:
        fig.add_subplot(rows, cols, pos+1)
        plt.imshow(x[index])
        plt.title("#" + str(index) + "-" + classes[y[index]])
        fig.subplots_adjust(hspace=1)
        pos+=1
    plt.savefig('Data_Stats/graphs/samples_training_img.png')
    plt.show()
        
#A function that changes the type of value obtained from 3d array to 1d   
def reshape_label(Y):
    Y = Y.reshape(-1,)
    return Y

#normalize the rgb data to values between 0-1:
def rgb_nor(X):
    return X/255

#A function that accepts a defined part according to indexes within the existing database (X_train / X_test) 
#and divides it into batchess arranged according to the client request.
def create_batches(X_train, Y_train, start_point, end_point):
    df = pd.DataFrame(list(zip(X_train[start_point:end_point], Y_train[start_point:end_point])), columns =['Image', 'label'])
    return df

#A function that receives a batch to it and the name of a column from which we issue
#the statistics on the data, the function issues a graph to each batch that enters with the amount of existing labels
#in the specific batch that enters it saves it locally and prints it to the client.
#The definition of i = 5 in the function itself in order to create generics
#in the spelling of the name of the locally saved file, (so that file number 6 is of X_test)
def batch_stat_plot(batch, column, i=5):
    sns.set(rc = {'figure.figsize':(15,10)})
    sns.countplot(x=batch[column], data=batch)
    filename ='Data_Stats/graphs/count of labels in batch'+ str(i+1)+ '.png'
    plt.savefig(filename)
    plt.show()
    
#A function that receives a batch and a  batch number, the function performs printing of the statistics that are also graphically issued during the program.
#The function receives a number only for the purpose of generically writing the batch name in print   
def print_batch_stat(batch, batch_num):
    print(f"stats for batch # {batch_num}")
    sum_sampels = sum(batch["label_name"].value_counts())
    print(f" # of sampels in batch - {sum_sampels}")
    print(batch["label_name"].value_counts())
    
#A function that receives a sample of an image and divides it within one graph into three 
#different images displayed in RGB colors.
#The function also receives a number in order to differentiate between the two samples that
#are stored locally on the computer (batch_sample, X_test sample).  
def rgb_splitter(sample_image, j):
    rgb_list = ['Reds','Greens','Blues']
    fig, ax = plt.subplots(1, 3, figsize=(15,5), sharey = True)
    for i in range(3):
        ax[i].imshow(sample_image[:,:,i], cmap = rgb_list[i])
        ax[i].set_title(rgb_list[i], fontsize = 15)
    plt.savefig('Data_Stats/graphs/sample_RGB_img'+str(j)+'.png')
    plt.show()

#A function that receives a sample of an image and performs a series of actions in order to extract statistics on it.
#1. Preserve the minimum and maximum value of the img array that holds within it the RGB values of the image sample.
#2. Save the image type and shape
#3. Label the image and its location.
#4.Rgb_splitter function call - to display a graph of the image in RGB colors
#5. Issue a histogram to the RGB values in the image sample
#6.Local retention of all data
#The function also receives a number in order to differentiate between the two samples that go out and are stored locally
#on the computer (batch_sample, X_test sample)   
def sample_stat(batch, index, i=2):
    min_rgb=batch.loc[index]["Image"].min()
    max_rgb=batch.loc[index]["Image"].max()
    img_shape = batch.loc[index]["Image"].shape
    label_name=batch.loc[index]["label_name"]
    sample_img=batch.loc[index]["Image"]
    print(f"Example of image {index} :")
    print(f" Min rgb value: {min_rgb}, Max rgb value: {max_rgb}")
    print(f"Img shape: {img_shape}")
    print(f"label: {label_name}")
    plt.imshow(sample_img)
    plt.savefig('Data_Stats/graphs/sample_img'+str(i)+'.png')
    plt.show()
    print("RBG split image:")
    rgb_splitter(sample_img, i)
    print("rgb values hist:")
    img_arr=np.array(batch_1.loc[index]['Image'])
    img_arr = reshape_label(img_arr)
    sns.set()
    sns.histplot(data=img_arr, kde=True)
    plt.savefig('Data_Stats/graphs/histplotfor RGB values'+str(i)+'.png')
    plt.show()

#A function that accepts a batch and issues that batch as a DF where the label 
#attribute is characterized as one_hot_vector
def one_hot_en(batch):
    dummies_df=pd.get_dummies(batch.label_name)
    merged_df=pd.concat([batch, dummies_df], axis=1)
    return merged_df

#A function that accepts a batch and the array of classes, the function adds to the batch a column called label_name,
#corresponding to the label number,
#The function returns the batch after the change    
def add_label_name(batch, classes):
    for index in batch.index:
        batch.at[index, 'label_name']=classes[batch.loc[index][ 'label']]
    return batch

#A function that gets a batch and finds for each row (for each image that exists within the batch), its minimum and maximum rgb values.
#The function loads the data into two new columns (two new attributes for DS) in the batch sent to it and returns it after the change. 
def min_max_rgb(batch):
    for index in batch.index:
        batch.at[index, 'min_rgb']=batch.loc[index]["Image"].min()
        batch.at[index, 'max_rgb']=batch.loc[index]["Image"].max()
    return batch

#A function that gets a batch and finds for each row (for each image that exists within the batch), the average rgb values in it.
#The function loads the data into three new columns (three new attributes for DS - mean_R,mean_G,mean_B) in the batch sent to it and returns it after the change.
def mean_r_g_b(batch):
    for index in batch.index:
        img=batch.at[index, 'Image']
        R = img[:,:,0] 
        G = img[:,:,1] 
        B = img[:,:,2]
        batch.at[index, 'mean_R']=np.mean(R)
        batch.at[index, 'mean_G']=np.mean(G)
        batch.at[index, 'mean_B']=np.mean(B)
    return batch

# A function that receives DF and issues a new DF where the
# X train values of the model are present
def prepare_x(batch):
    X_batch=batch[['Image', 'min_rgb', 'max_rgb', 'mean_R', 'mean_G', 'mean_B']]
    return X_batch

# A function that receives DF and issues a new DF where the
# Y train values of the model are present
def prepare_y(batch):
    Y_batch=batch[['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']]
    return Y_batch

# A function that calls the initialization of ANN_att model and
# returns it so that it receives the variable that holds the initialized model
def get_ANN_att_model():
    model = keras.Sequential([
            keras.layers.Dense(3000, input_shape=(3077,), activation='relu'),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(10, activation='sigmoid')    
        ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy','Precision','Recall'])
    return model

# A function that calls the initialization of ANN model and
# returns it so that it receives the variable that holds the initialized model
def get_ANN_model():
    model = keras.Sequential([
            keras.layers.Flatten(input_shape=(32,32,3)),
            keras.layers.Dense(3000, activation='relu'),
            keras.layers.Dense(1000, activation='relu'),
            keras.layers.Dense(10, activation='sigmoid')    
        ])

    model.compile(optimizer='SGD',
                  loss='categorical_crossentropy',
                  metrics=['accuracy','Precision','Recall'])
    return model

# A function that calls the initialization of CNN model and
# returns it so that it receives the variable that holds the initialized model   
def get_CNN_model():
    model = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
    
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
    
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
    
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision','Recall'])
    
    return model

# A function that performs a time calculation to calculate the model and issue results,
# The function receives the training data of the model and the
# amount of repetitions (error corrections),
# which it performs and returns the amount of time
# (in seconds) it took for the model to run
# The function is called after initial learning of the model in order to
# get the fastest run times for each model tested
def model_time(model, X_train, Y_train, epochs):
    start=time.time()
    model.fit(X_train, Y_train, epochs=epochs)
    end=time.time()
    
    return (end-start)



'''Data Exploration + normalization + visualization'''
      
if __name__ == '__main__':
    ''''Data Exploration'''
    #Loading the data:
    (X_train, Y_train), (X_test, Y_test) = datasets.cifar10.load_data()
    X_train.shape

    #Define img classes:
    classes = ["airplane", "automobile", "bird" , "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    #Shaping data
    Y_train=reshape_label(Y_train)
    Y_test=reshape_label(Y_test)
    
    #exploring data
    sample_plot(X_train, Y_train, classes, 0, 10)
    
    ''''Data normalization'''
    #normilazing the data:
    #1.normalize all training sets to 0-1 values
    X_train = rgb_nor(X_train)
    X_test = rgb_nor(X_test)
    
    #2.working on stats for training data:
    #2.a.divide all trainig sets to 10k batches
    batch_1=create_batches(X_train, Y_train, 0, 10000)
    batch_2=create_batches(X_train, Y_train, 10000, 20000)
    batch_3=create_batches(X_train, Y_train, 20000, 30000)
    batch_4=create_batches(X_train, Y_train, 30000, 40000)
    batch_5=create_batches(X_train, Y_train, 40000, 50000)
    batches_list=[batch_1,batch_2,batch_3,batch_4,batch_5]
   
    #2.b.adds label_name column:    
    for batch in batches_list:
        batch=add_label_name(batch, classes)
        
    #2.c.working on stats for test data:
    X_test_batch=create_batches(X_test, Y_test, 0, 10000)
      
    #2.d.adds label_name column:
    X_test_batch=add_label_name(X_test_batch, classes)
    
        
    ''''Data visualization'''
    #1.a.Add a LABEL column to each BATCH
    # && visulazing the count stsats for each batch in the training set:
    for i, batch in enumerate(batches_list): 
        batch_stat_plot(batch, 'label_name',i)
        print_batch_stat(batch, (i+1))
           
    #1.b.sample test for the 5850 indexd img in batch 1: 
    sample_stat(batch_1, 5850, 1)
    
    #2.a.visulazing the count stsats for test set: 
    batch_stat_plot(X_test_batch, 'label_name')
    print_batch_stat(X_test_batch, 1)
        
    #2.b.sample test for the first indexd img in test batch: 
    sample_stat(X_test_batch, 0)
    
    
    '''Data Preparation'''
    
    #1.one hot encoding for training batches && test batch:
    hot_batches_list=[]
    for batch in batches_list:    
        hot_batch = one_hot_en(batch)
        hot_batches_list.append(hot_batch)
    hot_X_test = one_hot_en(X_test_batch)

    #continue to run the same script for initilazing new attributes for the datat we have proccsed:
    #the data that was proccesed until this part have 2 main attributes(label, label_name) for every img (row) in the ds,
    #we will add 5 more importent attributes for each proccesed dataset
    #a.min_rgb
    #b.max_rgb
    #both executed and updating values to batches by min_max function
    #c.mean R
    #d.mean G
    #e.mean B
    #all are executed and updating values to batches by mean_r_g_b function
   
    #2.Train prepretion:
    #Add the attributes for each BATCH and output a csv that holds the df batch 
    #to a folder
    for i, batch in enumerate(hot_batches_list):
        batch=min_max_rgb(batch)
        batch=mean_r_g_b(batch)
        batch.to_csv('DF_to_csv/csv/proccesed hot batch'+str(i+1)+' with attributes.csv')
    
    #Union all BATCHES with attributes into one DF and issue it to a CSV file
    union_batch = pd.concat([hot_batches_list[0],hot_batches_list[1],hot_batches_list[2],hot_batches_list[3],hot_batches_list[4]], ignore_index=True)
    union_batch.to_csv('DF_to_csv/csv/proccesed hot union train batch.csv')
    
    #Division of the unified DF into X training values and Y training values
    #issue it to a CSV file
    X_train_batch=prepare_x(union_batch)
    X_train_batch.to_csv('DF_to_csv/csv/proccesed X_tarin_batch.csv')
    Y_train_batch=prepare_y(union_batch)
    Y_train_batch.to_csv('DF_to_csv/csv/proccesed Y_tarin_batch.csv')
    
    #Flattening the data (from the shape of a three-dimensional array that holds images in size 32 * 32 * 3),
    #to records that each record constitutes the
    #values of the image itself - hence a record in size 3072 values
    X_train=X_train.reshape(50000,3072)
    
    #Acquisition of the existing attributes in union_batch to a new variable and unification of the variable that
    #holds the X training values to one DF that holds within it the image values as records that consist of 3076 entries 
    #(after adding the attributes),Issuing the first five lines of the DF to CSV to display results
    X_train_att=X_train_batch.iloc[:,1:6].values
    clean_X_train=np.hstack((X_train,X_train_att))
    clean_X_train_df=pd.DataFrame(clean_X_train)
    clean_X_train_df_head=pd.DataFrame(clean_X_train_df.head())
    clean_X_train_df_head.to_csv('DF_to_csv/csv/proccesed clean_X_train_head.csv')
    
    #3.Test prepretion:
    #Repeat the procedure from the previous section to the X test values
    hot_X_test=min_max_rgb(hot_X_test)
    hot_X_test=mean_r_g_b(hot_X_test)
    hot_X_test.to_csv('DF_to_csv/csv/hot_X_test with attributes.csv')
    
    X_test_batch_split=prepare_x(hot_X_test)
    X_test_batch_split.to_csv('DF_to_csv/csv/proccesed X_test_batch.csv')
    
    X_test=X_test.reshape(10000,3072)
    X_test_att=X_test_batch_split.iloc[:,1:6].values
    clean_X_test=np.hstack((X_test,X_test_att))
    clean_X_test_df=pd.DataFrame(clean_X_test)
    clean_X_test_df_head=pd.DataFrame(clean_X_test_df.head())
    clean_X_test_df_head.to_csv('DF_to_csv/csv/proccesed clean_X_test_head.csv')
    
    #4.reshape the X values that get to the model - no attributes models:
    X_train_no_att=X_train.reshape(50000,32,32,3)
    X_test_no_att=X_test.reshape(10000,32,32,3)
    
    '''Models build & train'''
    '''ANN model build & train'''
    
    #initilazing the model
    ANN_model=get_ANN_model()
    #add model inputs + evaluation stats:
    ANN_model.fit(X_train_no_att, Y_train_batch, epochs=50)
    
    '''ANN model predictions:'''
    ANN_predictions=[]
    Y_test_truth=[]
    for i in range(10):
        ANN_predictions.append(classes[np.argmax(ANN_model.predict(X_test_no_att)[i])])
        Y_test_truth.append(classes[Y_test[i]])
        
    print(f'ANN  first 10 predictions: {ANN_predictions}')
    print(f'Y test first 10 GT: {Y_test_truth}')
    
    ''''ANN evaluation stats'''
    y_pred = ANN_model.predict(X_test_no_att)
    y_pred_classes = [np.argmax(element) for element in y_pred]

    print("Classification Report: \n", classification_report(Y_test, y_pred_classes))
    


    '''ANN_att model build & Train'''
    #initilazing the model
    ANN_att_model=get_ANN_att_model()
    
    #add model inputs + evaluation stats:
    ANN_att_model.fit(clean_X_train, Y_train_batch, epochs=50)
    
    
    '''ANN_att model predictions:'''
    ANN_att_predictions=[]
    Y_test_truth=[]
    for i in range(10):
        ANN_att_predictions.append(classes[np.argmax(ANN_att_model.predict(clean_X_test)[i])])
        Y_test_truth.append(classes[Y_test[i]])
        
    print(f'ANN att first 10 predictions: {ANN_att_predictions}')
    print(f'Y test first 10 GT: {Y_test_truth}')
    
    ''''ANN_att evaluation stats'''
    y_pred = ANN_att_model.predict(clean_X_test)
    y_pred_classes = [np.argmax(element) for element in y_pred]

    print("Classification Report: \n", classification_report(Y_test, y_pred_classes))
    
    
    '''CNN model build & Train''' 
    CNN_model=get_CNN_model()
    CNN_model.fit(X_train_no_att, Y_train_batch, epochs=50)
     
    '''model predictions:'''
    CNN_predictions=[]
    Y_test_truth=[]
    for i in range(10):
        CNN_predictions.append(classes[np.argmax(CNN_model.predict(X_test_no_att)[i])])
        Y_test_truth.append(classes[Y_test[i]])
        
    print(f'CNN first 10 predictions: {CNN_predictions}')
    print(f'Y test first 10 GT: {Y_test_truth}')
    
    ''''evaluation stats'''
    y_pred = CNN_model.predict(X_test_no_att)
    y_pred_classes = [np.argmax(element) for element in y_pred]

    print("Classification Report: \n", classification_report(Y_test, y_pred_classes))
    
    
    '''Time evaluation test to all models'''
    #comparing time models 10-50 epoches:
    #10 epoches:
    ANN_time_10_epoches = model_time(ANN_model, X_train_no_att, Y_train_batch, epochs=10)
    ANN_att_time_10_epoches = model_time(ANN_att_model, clean_X_train, Y_train_batch, epochs=10)
    CNN_time_10_epoches = model_time(CNN_model, X_train_no_att, Y_train_batch, epochs=10)
    print(f'10 epoches models time  --->  ANN : {ANN_time_10_epoches: .5f} , ANN_att : {ANN_att_time_10_epoches: .5f} , CNN : {CNN_time_10_epoches: .5f}')
    
    #50 epoches:
    ANN_time_50_epoches = model_time(ANN_model, X_train_no_att, Y_train_batch, epochs=50)
    ANN_att_time_50_epoches = model_time(ANN_att_model, clean_X_train, Y_train_batch, epochs=50)
    CNN_time_50_epoches = model_time(CNN_model, X_train_no_att, Y_train_batch, epochs=50)
    print(f'50 epoches models time  --->  ANN : {ANN_time_50_epoches: .5f} , ANN_att : {ANN_att_time_50_epoches: .5f} , CNN : {CNN_time_50_epoches: .5f}')
    

