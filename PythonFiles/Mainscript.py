# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



def LoadExcelFile(FileName='dataset.xlsx',FileDirectory='C:/Users/tancr/Desktop/RageQuit/'):
    
    import xlrd 
    
    loc = 'C:/Users/tancr/Desktop/RageQuit/' + FileName
      
    wb = xlrd.open_workbook(loc) 
    sheet = wb.sheet_by_index(0) 
    sheet.cell_value(0, 0) 
    
    DataSetFeatures = []
    DataSetLabels = []
    
    for i in range(sheet.nrows): 
        DataSetFeatures.append(sheet.cell_value(i, 1)) 
        DataSetLabels.append(sheet.cell_value(i, 2))
        
    return DataSetFeatures,DataSetLabels

def PreprocessFeatureSet(DataSetFeatures):
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    vectorizer = CountVectorizer(min_df=0, lowercase=False) 
    vectorizer.fit(DataSetFeatures)
    vectorizer.vocabulary_
    
    DataSetBagofWords = vectorizer.transform(DataSetFeatures).toarray()
    
    return DataSetBagofWords, vectorizer

def splitTrainingTesting(DataSet,OriginalFeatures,Labels,percentage,randomFlag):
    
    import numpy
    import keras
    
    if randomFlag == 1:
        tempSet = list(zip(DataSet,OriginalFeatures,Labels))
        numpy.random.shuffle(tempSet)
        DataSetNew,OriginalFeaturesNew,LabelsNew = zip(*tempSet)
    else:
        DataSetNew = DataSet
        OriginalFeaturesNew = OriginalFeatures
        LabelsNew = Labels
        
    lengDataSet = len(LabelsNew)
    
    trainIdx = round(lengDataSet*percentage)
        
    TrainingSet = numpy.array(DataSetNew[:trainIdx])
    TrainingLabels = numpy.array(LabelsNew[:trainIdx])
    OriginalFeaturesTrain = OriginalFeatures[:trainIdx]
    
    TestSet = numpy.array(DataSetNew[trainIdx:])
    TestLabels = numpy.array(LabelsNew[trainIdx:])
    OriginalFeaturesTest = OriginalFeatures[trainIdx:]
    
    TrainingLabels = keras.utils.to_categorical(TrainingLabels, 3)
    TestLabels = keras.utils.to_categorical(TestLabels, 3)
    
    return TrainingSet.reshape((TrainingSet.shape[0],TrainingSet.shape[1],1)),\
    TrainingLabels,\
    TestSet.reshape((TestSet.shape[0],TestSet.shape[1],1)),\
    TestLabels,\
    OriginalFeaturesTrain,\
    OriginalFeaturesTest
    
#    return TrainingSet.reshape((TrainingSet.shape[0],TrainingSet.shape[1],1)),\
#    TrainingLabels.reshape((TrainingLabels.shape[0],TrainingLabels.shape[1],1)),\
#    TestSet.reshape((TestSet.shape[0],TestSet.shape[1],1)),\
#    TestLabels.reshape((TestLabels.shape[0],TestLabels.shape[1],1))

#    return numpy.array(TrainingSet).reshape((TrainingSet.shape[0],TrainingSet.shape[1],1)),\
#        numpy.array(TrainingLabels).reshape((TrainingSet.shape[0],TrainingSet.shape[1],1)),\
#        numpy.array(TestSet).reshape((TestSet.shape[0],TestSet.shape[1],1)),\
#        numpy.array(TestLabels).reshape((TestSet.shape[0],TestSet.shape[1],1))
    
def CreateOutput(TestSet,Full_model,OriginalCommentsTest):
    
    import numpy as np
    
    output = Full_model.predict(TestSet)
    
    idx = np.argmax(output, axis=-1)
    idx[idx == 2] = -1

    outputComments = []
    outputScores = []

    for i in range(len(OriginalCommentsTest)):
        outputComments.append(OriginalCommentsTest[i])
        outputScores.append(idx[i])
        
        
    return outputComments,outputScores

def WriteCsvOutputFile(outputComments,outputScores,filename):
    
    import csv
    from itertools import zip_longest
    d = [outputComments, outputScores]
    export_data = zip_longest(*d, fillvalue = '')
    with open(filename + '.csv', 'w', newline='') as myfile:
          wr = csv.writer(myfile)
          wr.writerows(export_data)
    myfile.close()
    
   
def model(inputShape):
    
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    
    inputLayer = Input(inputShape)
    Flat = Flatten()(inputLayer)
    Dense1 = Dense(256, activation='relu')(Flat)
    Drop = Dropout(0.25)(Dense1)
    Dense2 = Dense(64, activation='relu')(Drop)
    Drop2 = Dropout(0.15)(Dense2)
    Out = Dense(3,activation='softmax')(Drop2)
    
    model = Model(inputLayer,Out)
    
    return model

def Mainscript():
    
    import keras
    from keras.optimizers import SGD
    
    ''' Parameters of the CNN '''
    num_epochs = 100
    batch_sz = 16
#    num_classes = 3
    lrate = 0.001
    decay_rate = 1e-6
    moment = 0.9
    loss_fun = 'categorical_crossentropy'

    ''' Load Excel File '''
    Features, Labels = LoadExcelFile()
    
    ''' PreProcess Features '''
    
    FeaturesPP, vectorizer = PreprocessFeatureSet(Features)
    
    ''' Make Training and Testing Set '''
    TrainingSet,TrainingLabels,TestSet,TestLabels,OriginalCommentsTrain,OriginalCommentsTest\
    = splitTrainingTesting(FeaturesPP,Features,Labels,0.70,1)
    
    ''' Creation of Model '''
    Full_model = model((TrainingSet.shape[1],1))
    
    sgd = SGD(lr= lrate, decay= decay_rate, momentum=moment, nesterov=True)
    Full_model.compile(loss=loss_fun,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    ''' Training of Model '''
    Full_model.fit(TrainingSet, TrainingLabels,
          batch_size=batch_sz,
          epochs=num_epochs,
          verbose=1,
          validation_data=(TestSet,TestLabels))
    
    ''' Evaluation of Model '''
    score = Full_model.evaluate(TestSet, TestLabels, verbose=0)
    
    ''' Create Output List '''
    outputComments,outputScores = CreateOutput(TestSet,Full_model,OriginalCommentsTest)
    
    return outputComments,outputScores

if __name__ == "__main__":
    Comments,Scores = Mainscript()
    WriteCsvOutputFile(Comments,Scores,'SentimentAnalysis')


