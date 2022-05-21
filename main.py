import tkinter as tk
from tkinter import NW, YES, BOTH
from pandas import read_csv
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

col_names = ['title', 'text', 'subject', 'date', 'True or Fake']
dataset = read_csv('True and Fake.csv', header=None, names=col_names)

root = tk.Tk()
root.title('Fake News Detector')
bg_img = tk.PhotoImage(file="bg4.png")
canvas1 = tk.Canvas(root, width=600, height=600, relief='groove')
canvas1.pack(expand=YES, fill=BOTH)
canvas1.create_image(10, 10, image=bg_img, anchor=NW)

label1 = tk.Label(root, text="Fake News Detector", bg='black', fg='white')
label1.config(font=('Times', 20, 'bold'))
canvas1.create_window(300, 30, window=label1)


def PrintDatasetWindow():
    window1 = tk.Tk()
    canvas2 = tk.Canvas(window1, width=600, height=400, relief='groove')
    canvas2.grid()
    label2 = tk.Label(window1, text=dataset)
    label2.config(font=('Times', 10, 'bold'))
    canvas2.create_window(280, 180, window=label2)


img1 = tk.PhotoImage(file="button_print-the-dataset (2).png")
PrintDatasetButton = tk.Button(image=img1, command=PrintDatasetWindow, border='0', bg='black')
canvas1.create_window(300, 100, window=PrintDatasetButton)


def DataPreprocessingWindow():
    le = preprocessing.LabelEncoder()
    dataset['title'] = le.fit_transform(dataset['title'].astype(str))
    dataset['text'] = le.fit_transform(dataset['text'].astype(str))
    dataset['subject'] = le.fit_transform(dataset['subject'].astype(str))
    dataset['date'] = le.fit_transform(dataset['date'].astype(str))
    new_window = tk.Tk()
    new_window.title('preprocessing successful')
    new_window.geometry('300x300')
    label2 = tk.Label(new_window, text='Preprocessing Successful !', relief='groove',
                      font=('Times', 12, 'bold')).place(x=30, y=70)


img2 = tk.PhotoImage(file="button_preprocess-the-data (1).png")
ProcessDataButton = tk.Button(image=img2, command=DataPreprocessingWindow, bg='black', border='0')

canvas1.create_window(300, 170, window=ProcessDataButton)


def SplitDataWindow():
    SplitWindow = tk.Tk()
    SplitWindow.title('Split dataset')
    new_canvas = tk.Canvas(SplitWindow, width=400, height=400, relief='groove')
    new_canvas.pack()
    label2 = tk.Label(SplitWindow, text='Enter test size')
    label2.config(font=('Times', 10, 'bold'))
    new_canvas.create_window(200, 100, window=label2)
    global entry_split
    entry_split = tk.Entry(SplitWindow)
    new_canvas.create_window(200, 150, window=entry_split)
    SplitButton = tk.Button(SplitWindow, text='Split dataset', command=applySplit, fg='white', bg='green', border='1')
    new_canvas.create_window(200, 200, window=SplitButton)


def applySplit():
    window6 = tk.Tk()
    window6.title('split dataset')
    canvas6 = tk.Canvas(window6, width=400, height=400, relief='groove')
    canvas6.pack()
    feature_cols = ['title', 'text', 'subject', 'date']
    label = ['True or Fake']
    x = dataset[feature_cols]
    y = dataset[label]
    global x_train, x_test, y_train, y_test
    testSize = float(entry_split.get())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testSize, random_state=3)
    label_split = tk.Label(window6, text='The dataset has been split into a training set and test set',
                           relief='groove')
    label_split.config(font=('Times', 10, 'bold'))
    canvas6.create_window(200, 50, window=label_split)


img3 = tk.PhotoImage(file="button_split-the-dataset.png")
SplitDataButton = tk.Button(image=img3, command=SplitDataWindow, bg='black', border='0')
canvas1.create_window(300, 240, window=SplitDataButton)


def DecisionTreeClassifier1():
    window2 = tk.Tk()
    window2.title('Decision Tree')
    canvas2 = tk.Canvas(window2, width=200, height=200, relief='groove')
    canvas2.pack()
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    label_accuracy1 = tk.Label(window2, text='Accuracy')
    label_accuracy1.config(font=('Times', 20, 'bold'), bg='green', fg='white')
    canvas2.create_window(100, 50, window=label_accuracy1)
    accuracy = metrics.accuracy_score(y_test, y_predict) * 100
    label_accuracy2 = tk.Label(window2, text=accuracy)
    canvas2.create_window(100, 80, window=label_accuracy2)


img4 = tk.PhotoImage(file="button_apply-decision-tree-classifier (1).png")
DecisionTreeClassifierButton = tk.Button(image=img4, command=DecisionTreeClassifier1, bg='black', border='0')

canvas1.create_window(300, 310, window=DecisionTreeClassifierButton)

img5 = tk.PhotoImage(file="button_apply-kn-neighbors.png")


def KnnWindow():
    window3 = tk.Tk()
    window3.title('Knn')
    canvas3 = tk.Canvas(window3, width=400, height=400, relief='groove')
    canvas3.pack()
    label_enter_neighbors = tk.Label(window3, text="Enter number of neighbours")
    canvas3.create_window(200, 100, window=label_enter_neighbors)
    global entry_neighbors
    entry_neighbors = tk.Entry(window3)
    canvas3.create_window(200, 150, window=entry_neighbors)
    buttonApplyKnn = tk.Button(window3, text='Apply knn', command=applyKnn, bg='green', border='1', fg='white')
    canvas3.create_window(200, 250, window=buttonApplyKnn)


KNeighborsButton = tk.Button(image=img5, command=KnnWindow, bg='black', border='0')
canvas1.create_window(300, 380, window=KNeighborsButton)


def applyKnn():
    window4 = tk.Tk()
    window4.title('KNeighbors')
    canvas4 = tk.Canvas(window4, width=200, height=200, relief='groove')
    canvas4.pack()
    numberOfNeighbors = int(entry_neighbors.get())
    knn = KNeighborsClassifier(n_neighbors=numberOfNeighbors)
    knn.fit(x_train, y_train.values.ravel())
    y_predict_knn = knn.predict(x_test)
    knn_accuracy = metrics.accuracy_score(y_test, y_predict_knn) * 100
    label_accuracy1 = tk.Label(window4, text='Accuracy')
    label_accuracy1.config(font=('Times', 20, 'bold'), bg='green', fg='white')
    canvas4.create_window(100, 50, window=label_accuracy1)
    label_accuracy2 = tk.Label(window4, text=knn_accuracy)
    canvas4.create_window(100, 100, window=label_accuracy2)


def RandomForrest():
    window4 = tk.Tk()
    window4.title('Random Forrest')
    canvas4 = tk.Canvas(window4, width=400, height=400, relief='groove')
    canvas4.pack()
    label_estimator = tk.Label(window4, text='Enter number of estimators')
    label_estimator.config(font=('Times', 10, 'bold'))
    canvas4.create_window(100, 100, window=label_estimator)
    global entry_estimators
    entry_estimators = tk.Entry(window4)
    canvas4.create_window(100, 150, window=entry_estimators)
    randomForrestButton = tk.Button(window4, text='Apply Random Forrest', command=applyRandomForrest
                                    , fg='white', bg='green', border='1')
    canvas4.create_window(100, 200, window=randomForrestButton)


def applyRandomForrest():
    window7 = tk.Tk()
    window7.title('Random Forrest')
    canvas7 = tk.Canvas(window7, width=200, height=200)
    canvas7.pack()
    estimators = int(entry_estimators.get())
    clf = RandomForestClassifier(n_estimators=estimators)
    clf = clf.fit(x_train, y_train.values.ravel())
    y_predict_random_forest = clf.predict(x_test)
    RandomForrestAccuracy = metrics.accuracy_score(y_test, y_predict_random_forest) * 100
    label_accuracy1 = tk.Label(window7, text='Accuracy')
    label_accuracy1.config(font=('Times', 20, 'bold'), bg='green', fg='white')
    canvas7.create_window(100, 100, window=label_accuracy1)
    label_accuracy2 = tk.Label(window7, text=RandomForrestAccuracy)
    canvas7.create_window(100, 150, window=label_accuracy2)


img6 = tk.PhotoImage(file="button_apply-random-forrest.png")
KNeighborsButton = tk.Button(image=img6, command=RandomForrest, bg='black', border='0')
canvas1.create_window(300, 450, window=KNeighborsButton)

root.mainloop()
