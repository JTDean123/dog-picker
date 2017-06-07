# --------------------------------------------
#
# Dog breed classification based on an image
# LR and CNN
# Jason Dean
# June 8th, 2017
# jtdean@gmail.com
#
# --------------------------------------------

import logging
import time
import os, os.path
from PIL import Image
import numpy as np
np.random.seed(123)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')


# --------------- functions ---------------

def logger():
    # initialize logging
    LOG_FILENAME = 'dog_picker_' + time.strftime("%d%m%Y") + '.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    logging.info('Dog Breed Classification')
    start_mssg = "Program Started: " + time.strftime("%d/%m/%Y") + " " + str(time.strftime("%H:%M:%S"))
    logging.info(start_mssg)


def image_info(paths, image_names):
    # determine the dimensions (pixels) of the doggie images
    length = 0
    for i in image_names:
        length += len(i)

    sizes = pd.DataFrame(index=range(length),
                         columns=['ratio', 'height', 'width', 'breed'])

    breeds = ['West_Highland_white_terrier',
              'Irish_wolfhound',
              'black_tan_coonhound']

    counter = 0
    for i, j in enumerate(image_names):
        for y in j:
            file_name = str(paths[i])
            file = file_name + str(y)
            w, h = Image.open(file).size
            sizes['ratio'].iloc[counter] = w / h
            sizes['width'].iloc[counter] = w
            sizes['height'].iloc[counter] = h
            sizes['breed'].iloc[counter] = str(breeds[i])
            counter = counter + 1

    return (sizes)


def get_dimensions(image_sizes):
    # calculate descriptive statistics for width and height for each breed
    dimensions = pd.DataFrame(index=('West_Highland_white_terrier',
                                     'Irish_wolfhound',
                                     'black_tan_coonhound'),
                              columns=['width', 'height'])

    dimensions['width'].loc['West_Highland_white_terrier'] = image_sizes['width'][image_sizes['breed']=='West_Highland_white_terrier'].mean()
    dimensions['height'].loc['West_Highland_white_terrier'] = image_sizes['height'][image_sizes['breed']=='West_Highland_white_terrier'].mean()

    dimensions['width'].loc['Irish_wolfhound'] = image_sizes['width'][image_sizes['breed']=='Irish_wolfhound'].mean()
    dimensions['height'].loc['Irish_wolfhound'] = image_sizes['height'][image_sizes['breed']=='Irish_wolfhound'].mean()

    dimensions['width'].loc['black_tan_coonhound'] = image_sizes['width'][image_sizes['breed']=='black_tan_coonhound'].mean()
    dimensions['height'].loc['black_tan_coonhound'] = image_sizes['height'][image_sizes['breed']=='black_tan_coonhound'].mean()

    return dimensions


def resizer(paths, image_names):
    # convert to black and white and resize and covert to 1Xpixel vector
    image_features = []

    breeds = ['West_Highland_white_terrier',
              'Irish_wolfhound',
              'black_tan_coonhound']

    for i, j in enumerate(image_names):
        for y in j:
            file_name = str(paths[i])
            file = file_name + str(y)
            img = Image.open(file).convert('L')
            img = img.resize((450, 400))
            img = np.array(img)
            img = img.reshape(1, img.shape[0] * img.shape[1])
            image_features.append(img[0])

    return (image_features)


def features(labels):
    # create a feature vector
    y_labels = []

    for i in labels:
        if i == 'West_Highland_white_terrier':
            y_labels.append(0)
        elif i == 'Irish_wolfhound':
            y_labels.append(1)
        else:
            y_labels.append(2)

    return y_labels

def features_unit_test():
    practice = ['West_Highland_white_terrier',
                'Irish_wolfhound',
                'black_tan_coonhound']

    practice_labels = features(practice)
    output = [0,1,2]
    if output != practice_labels:
        print('failed feature label unit test')


def pca_LR(X_train, X_test, y_train, y_test):
    # perform PCA dimensionality reduction followed by logistic regression
    logging.info("Starting PCA and LR")
    # solver = 'auto' to handle the large number of features
    pca = PCA(n_components=400, svd_solver='auto')

    # perform PCA and project the original data on to the new coordinates
    X_train_PCA = pca.fit(X_train)
    X_test_PCA = pca.fit(X_test)

    X_train_PCA_fit = X_train_PCA.transform(X_train)
    X_test_PCA_fit = X_test_PCA.transform(X_test)

    # -------- plot the explained variance versus the number of PCs --------
    cumsum = X_train_PCA.explained_variance_ratio_.cumsum()

    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))

        font = {'weight': 'bold',
                'size': 18}

        plt.rc('font', **font)

        plt.plot(range(1, len(cumsum) + 1), cumsum, '-o', color='b')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Proportion')
        plt.tight_layout()
        plt.savefig('PCA.pdf')
        plt.rcdefaults()

        logging.info("Saved PCA plot as: \n PCA.pdf")
    # --------------------------------

    # build a logistic regression model
    multi_LR = LogisticRegression(multi_class='multinomial', max_iter=10000)

    # hyperparameter tuning via GridSearch
    parameter_candidates = [
        {'C': list(np.power(10.0, np.arange(-13, -7))), 'solver': ['sag', 'lbfgs', 'newton-cg']},
    ]

    clf = GridSearchCV(multi_LR, parameter_candidates, cv=10)
    clf.fit(X_train_PCA_fit, y_train)

    print('LR hyperparameter tuning results:  \n')
    print('Best C:', clf.best_estimator_.C)
    print('Best solver:', clf.best_estimator_.solver)
    lr_tune_log = "LR hyperparamter resuts: best C- " + str(clf.best_estimator_.C) + " best solver:  " + str(clf.best_estimator_.solver)
    logging.info(lr_tune_log)

    # run a logistic regression model using the best parameters
    multi_LR = LogisticRegression(multi_class='multinomial',
                                  max_iter=10000,
                                  C=clf.best_estimator_.C,
                                  solver=clf.best_estimator_.solver)

    multi_LR_fit = multi_LR.fit(X_train_PCA_fit, y_train)

    print("LR accuracy:  ", multi_LR_fit.score(X_test_PCA_fit, y_test))
    lr_acc_log = "LR accuracy:  " + str(multi_LR_fit.score(X_test_PCA_fit, y_test))
    logging.info(lr_acc_log)

    # make predictions using the LR model
    predictions = multi_LR_fit.predict(X_test_PCA_fit)

    # build a confusion matrix from the predictions
    confusion = pd.DataFrame(confusion_matrix(y_test, predictions))
    confusion.columns = ['Predicted West_Highland_white_terrier',
                         'Predicted Irish_wolfhound',
                         'Predicted black_tan_coonhound']

    confusion.index = ['Actual West_Highland_white_terrier',
                       'Actual Irish_wolfhound',
                       'Actual black_tan_coonhound']

    confusion.to_csv('LR_confusion_matrix.csv')
    logging.info('Saved LR confusion matrix to:  \n LR_confusion_matrix.csv')

    # determine model performance metrics
    target_names = ['West_Highland_white_terrier',
                    'Irish_wolfhound',
                    'black_tan_coonhound']

    print(classification_report(y_test, predictions, target_names=target_names))

    logging.info("Finished PCA and LR")



def resizer_CNN(paths, image_names):
    # resize and covert to 1Xpixel vector (keep colors)
    # image_features = np.zeros((546, 400, 450, 3))
    image_features = np.zeros((546, 96, 96, 3))

    breeds = ['West_Highland_white_terrier',
              'Irish_wolfhound',
              'black_tan_coonhound']

    counter = 0
    for i, j in enumerate(image_names):
        for y in j:
            file_name = str(paths[i])
            file = file_name + str(y)
            img = Image.open(file).convert()
            # img = img.resize((450,400))
            img = img.resize((96, 96))
            img = np.array(img)
            image_features[counter] = img
            counter = counter + 1

    return (image_features)


def CNN(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn):
    # build and run a CNN for the image classification
    logging.info('Starting CNN')

    # reshape the data
    X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], 3, 96, 96)
    X_test_cnn = X_test_cnn.reshape(X_test_cnn.shape[0], 3, 96, 96)

    y_train_cnn = np_utils.to_categorical(y_train_cnn, 3)
    y_test_cnn = np_utils.to_categorical(y_test_cnn, 3)

    X_train_cnn = X_train_cnn.astype('float32')
    X_test_cnn = X_test_cnn.astype('float32')

    X_train_cnn /= np.max(X_train_cnn)
    X_test_cnn /= np.max(X_test_cnn)

    # build a simple model that doesn't freak out my laptop
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(3, 96, 96), dim_ordering='th'))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1028, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    # compile the model
    #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # fit the model!
    logging.info('Training CNN with 35 epochs- go get a beverage, this can take a while....')
    time_cnn_start = 'CNN training start time:  ' + str(time.strftime("%H:%M:%S"))
    logging.info(time_cnn_start)

    model.fit(X_train_cnn, y_train_cnn, batch_size=16, nb_epoch=40, verbose=1)

    time_cnn_finish = 'CNN training finish time:  ' + str(time.strftime("%H:%M:%S"))
    logging.info(time_cnn_finish)

    # make predictions
    predict = model.predict_classes(X_test_cnn)

    # calcualate total accuracy
    actual = np.argmax(y_test_cnn, axis=1)
    right = 0
    for i, j in enumerate(predict):
        if j == actual[i]:
            right += 1
    accuracy_cnn = right / len(actual)

    print("CNN accuracy:  ", accuracy_cnn)
    cnn_output = "CNN accuracy:  " + str(accuracy_cnn)
    logging.info(cnn_output)

    # write predictions to a confusion matrix
    confusion_cnn = pd.DataFrame(confusion_matrix(actual, predict))

    confusion_cnn.columns = ['Predicted West_Highland_white_terrier',
                             'Predicted Irish_wolfhound',
                             'Predicted black_tan_coonhound']

    confusion_cnn.index = ['Actual West_Highland_white_terrier',
                           'Actual Irish_wolfhound',
                           'Actual black_tan_coonhound']

    confusion_cnn.to_csv('CNN_confusion_matrix.csv')
    logging.info('Saved CNN confusion matrix to:  \n CNN_confusion_matrix.csv')

# --------------- main ---------------
def run():

    # start logging
    logger()

    # determine the number of dogs in each breed
    try:
        West_Highland_white_terrier = os.listdir('Images/n02098286-West_Highland_white_terrier')
        Irish_wolfhound = os.listdir('Images/n02090721-Irish_wolfhound')
        black_tan_coonhound = os.listdir('Images/n02089078-black-and-tan_coonhound')
        print(" West_Highland_white_terrier:  ", len(West_Highland_white_terrier), '\n', 'Irish_wolfhound: ',
              len(Irish_wolfhound), '\n', 'black_tan_coonhound: ', len(black_tan_coonhound))
    except:
        logging.warning("Error: could not open images")


    # get the dimensions of the images
    images = [West_Highland_white_terrier, Irish_wolfhound, black_tan_coonhound]
    image_locs = ['Images/n02098286-West_Highland_white_terrier/',
                  'Images/n02090721-Irish_wolfhound/',
                  'Images/n02089078-black-and-tan_coonhound/']

    size = image_info(image_locs, images)


    #-------- plot size histograms --------
    font = {'weight': 'bold',
            'size': 22}

    plt.rc('font', **font)

    bins = list(np.arange(0, 1000, 25))

    plt.figure(figsize=(8, 5))
    size['width'].plot.hist(bins=bins, width=20, color='lightblue', edgecolor="black")
    plt.xlabel('width')
    plt.ylabel('count')
    plt.title('Image Widths')
    plt.savefig('image_width_hist.pdf')

    plt.figure(figsize=(8, 5))
    size['height'].plot.hist(bins=bins, width=20, color='green', edgecolor="black")
    plt.xlabel('height')
    plt.ylabel('count')
    plt.title('Image Heights')
    plt.savefig('image_height_hist.pdf')
    plt.rcdefaults()

    logging.info("Saved height and width histogram plots as: \n image_width_hist.pdf, image_height_hist.pdf")
    # --------------------------------

    logging.info('Starting image processing for logistic regression')

    # calculate average dimensions for each breed
    dims = get_dimensions(size)
    print('\nAverage image height and width: ')
    print(dims.head())

    # re-scale the images to a width of 450 and a height of 400.
    # also, we will convert them to back and white.
    # this will return a 1Xpixel vector for each image
    X = np.array(resizer(image_locs, images))
    logging.info("Converted images to 450 width, 400 height, and black and white")

    # create a feature vector for each breed
    features_unit_test()
    y = features(size['breed'])

    # split data into test and training sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=42)

    # perform PCA and logistic regression on the data
    pca_LR(Xtrain, Xtest, ytrain, ytest)

    # resize images for a CNN
    logging.info('Starting image processing for CNN')
    X_cnn = resizer_CNN(image_locs, images)

    # split the data up into test and training
    Xtrain_cnn, Xtest_cnn, ytrain_cnn, ytest_cnn = train_test_split(X_cnn, y, test_size=0.30, random_state=42)
    CNN(Xtrain_cnn, Xtest_cnn, ytrain_cnn, ytest_cnn)

    # goodbye!
    exit_message = 'Program finished:  ' + str(time.strftime("%H:%M:%S"))
    logging.info(exit_message)



# -------- go time --------
if __name__ == '__main__':
    run()
