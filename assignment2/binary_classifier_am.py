from PIL import Image
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    img = img.resize((64,64))
    data = np.asarray(img, dtype="int32")
    return data

def save_image(npdata, outfilename):
    img = Image.fromarray(np.asarray(
        np.clip(npdata, 0, 255), dtype="uint8"), "L")
    img.save(outfilename)

zeros = []
ones = []

data_dirs = ["../data/Train/0", "../data/Train/1"]

print("Beginning image reading")
for file in os.listdir(data_dirs[0]):
    print("Reading: {}".format(file))
    zeros.append(load_image(os.path.join(data_dirs[0],file)).flatten())

for file in os.listdir(data_dirs[1]):
    print("Reading: {}".format(file))
    ones.append(load_image(os.path.join(data_dirs[1],file)).flatten())

X = np.array(zeros + ones)

zero_labels = np.zeros((len(zeros),1))
one_labels = np.ones((len(ones),1))
y = np.ravel(np.vstack((zero_labels, one_labels)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Training classifier...")
clf = LogisticRegression(solver="lbfgs", verbose=True, n_jobs=-1).fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(score)




