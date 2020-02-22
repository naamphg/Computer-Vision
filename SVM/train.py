import cv2
import os
import pickle
import numpy as np
from imutils import paths
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def get_hog() :
    winSize = (32,32)
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (4,4)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    signedGradient = True

    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
    
    return hog
    affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR

print("[INFO] grab the paths to the input images in our dataset...")
imagePaths = list(paths.list_images('dataset'))

knownSigns = []
hog = get_hog()
print('Calculating HoG descriptor for every image ... ')
hog_descriptors = []

for (i, imagePath) in enumerate(imagePaths):
    #print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]    
    image = cv2.imread(imagePath, 0)
    blur = cv2.GaussianBlur(image,(3,3),0)
    ret, image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.resize(image,(32,32))
    hog_descriptors.append(hog.compute(image))
    knownSigns.append(name)
# =>we had hog_descriptors and knownSigns
# remove single dimension in hog_descriors    
hog_descriptors=np.squeeze(hog_descriptors)
#convert labels to numerical format
le=LabelEncoder()
labels=le.fit_transform(knownSigns)


hog_descriptors_train, hog_descriptors_test, labels_train, labels_test = train_test_split(hog_descriptors, labels, test_size = 0.2, random_state=0)
print("[INFO] training model...")
clf = svm.SVC(C = 100.0, kernel='rbf', gamma= 0.001,probability=True)
print(hog_descriptors_train.shape,labels_train.shape)
#train classifier
clf.fit(hog_descriptors_train, labels_train)

### Make predictions on test data
a=clf_predictions = clf.predict(hog_descriptors_test)
print('output',a)
score = accuracy_score(labels_test, clf_predictions)*100
print('classifier score: ',score)
cm = confusion_matrix(labels_test, clf_predictions)
print('confusion matrix:', cm)

### Grid Search
#print("Performing grid search ... ")
#param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
#clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
#clf_grid.fit(hog_descriptors_train, labels_train)
#print("Best Parameters:\n", clf_grid.best_params_)
#print("Best Estimators:\n", clf_grid.best_estimator_)

# write the classifier to disk
out = open("output/clf.pkl", "wb")
out.write(pickle.dumps(clf))
out.close()

# write the label encoder to disk
out = open("output/le.pkl", "wb")
out.write(pickle.dumps(le))
out.close()



