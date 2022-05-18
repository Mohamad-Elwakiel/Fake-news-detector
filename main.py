import tkinter as tk
from tkinter import NW, YES, BOTH
from pandas import read_csv
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

col_names = ['title', 'text', 'subject', 'date', 'True or Fake']
dataset = read_csv('True and Fake.csv', header=None, names=col_names)

root = tk.Tk()
root.title('Fake News Detector')
bg_img = tk.PhotoImage(file="bg2.png")
bg_img_label = tk.Label(root, image=bg_img)
canvas1 = tk.Canvas(root, width=600, height=600, relief='groove', bg='#c2c2c2')
canvas1.pack(expand=YES, fill=BOTH)
# bg_img_label.pack()

label1 = tk.Label(root, text="Fake News Detector", bg='#c2c2c2')
label1.config(font=('Times', 20, 'bold'))
canvas1.create_window(300, 30, window=label1)


def PrintDatasetWindow():
    window1 = tk.Tk()
    canvas2 = tk.Canvas(window1, width=400, height=400, relief='groove')
    canvas2.pack()
    label2 = tk.Label(window1, text=dataset)
    label2.config(font=('Times', 10, 'bold'))
    canvas2.create_window(80, 100, window=label2)


img1 = tk.PhotoImage(file="button_print-the-dataset (2).png")
PrintDatasetButton = tk.Button(text='Print Dataset', image=img1, command=PrintDatasetWindow, border='0', bg='#c2c2c2')
canvas1.create_window(80, 100, window=PrintDatasetButton)


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
ProcessDataButton = tk.Button(image=img2, command=DataPreprocessingWindow, bg='#c2c2c2', border='0')

canvas1.create_window(80, 170, window=ProcessDataButton)


def SplitDataWindow():
    # global test_size
    feature_cols = ['title', 'text', 'subject', 'date']
    label = ['True or Fake']
    x = dataset[feature_cols]
    y = dataset[label]
    SplitWindow = tk.Tk()
    new_canvas = tk.Canvas(SplitWindow, width=800, height=800, relief='groove')
    new_canvas.pack()
    label2 = tk.Label(SplitWindow, text='Test size is 0.2')
    label2.config(font=('Times', 28, 'bold'))
    new_canvas.create_window(400, 50, window=label2)
    # entry1 = tk.Entry(SplitWindow)
    # new_canvas.create_window(200, 160, window=entry1)
    # test_size = int(entry1.get())
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
    label5 = tk.Label(SplitWindow, text='y_test: ')
    label5.config(font=('Times', 10, 'bold'))
    new_canvas.create_window(100, 150, window=label5)
    label3 = tk.Label(SplitWindow, text=y_test)
    label3.config(font=('Times', 10))
    new_canvas.create_window(100, 200, window=label3)
    label6 = tk.Label(SplitWindow, text='x_test: ')
    label6.config(font=('Times', 10))
    new_canvas.create_window(100, 350, window=label6)
    label4 = tk.Label(SplitWindow, text=x_test)
    label4.config(font=('Times', 10))
    new_canvas.create_window(100, 400, window=label4)


img3 = tk.PhotoImage(file="button_split-the-dataset.png")
SplitDataButton = tk.Button(image=img3, command=SplitDataWindow, bg='#c2c2c2', border='0')
canvas1.create_window(80, 240, window=SplitDataButton)


def DecisionTreeClassifier1():
    window2 = tk.Tk()
    canvas2 = tk.Canvas(window2, width=200, height=200, relief='groove')
    canvas2.pack()
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    # label_predict = tk.Label(window2, text=y_predict)
    # label_predict.config(font=('Times', 10, 'bold'))
    # canvas2.create_window(100, 200, window=label_predict)
    label_accuracy1 = tk.Label(window2, text='Accuracy')
    label_accuracy1.config(font=('Times', 20, 'bold'), bg='green', fg='white')
    canvas2.create_window(100, 50, window=label_accuracy1)
    accuracy = metrics.accuracy_score(y_test, y_predict) * 100
    label_accuracy2 = tk.Label(window2, text=accuracy)
    canvas2.create_window(100, 80, window=label_accuracy2)


img4 = tk.PhotoImage(file="button_apply-decision-tree-classifier (1).png")
DecisionTreeClassifierButton = tk.Button(image=img4, command=DecisionTreeClassifier1, bg='#c2c2c2', border='0')
canvas1.create_window(360, 100, window=DecisionTreeClassifierButton)


def KnnWindow():
    window3 = tk.Tk()
    canvas3 = tk.Canvas(window3, width=400, height=400, relief='groove')
    canvas3.pack()
    label_neighbors = tk.Label(window3, text='Enter the number of neighbors')
    label_neighbors.config(font=('Times', 10))
    canvas3.create_window(100, 50, window=label_neighbors)
    entry_neighbors = tk.Entry(window3)
    canvas3.create_window(100, 80, window=entry_neighbors)
    # n_neighbors = int(entry_neighbors.get())

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train.values.ravel())
    y_predict_knn = knn.predict(x_test)
    knn_accuracy = metrics.accuracy_score(y_test, y_predict_knn) * 100
    label_accuracy1 = tk.Label(window3, text='Accuracy')
    label_accuracy1.config(font=('Times', 20, 'bold'), bg='green', fg='white')
    canvas3.create_window(100, 200, window=label_accuracy1)
    label_accuracy2 = tk.Label(window3, text=knn_accuracy)
    canvas3.create_window(100, 230, window=label_accuracy2)


img5 = tk.PhotoImage(file="button_apply-kn-neighbors.png")
KNeighborsButton = tk.Button(image=img5, command=KnnWindow, bg='#c2c2c2', border='0')
canvas1.create_window(360, 170, window=KNeighborsButton)
root.mainloop()
