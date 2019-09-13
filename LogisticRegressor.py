import numpy as np
import matplotlib.pyplot as plt
from tkinter import *

def final_variables(n_pts,sd1,sd2,iterations,alpha):
    def draw(x1, x2):
        ln = plt.plot(x1, x2,'-')
        plt.pause(0.0001)
        ln[0].remove()


    def sigmoid(score):
        return 1 / (1 + np.exp(-score))




    def gradient_descent(line_parameters, points, y, alpha):
        n = points.shape[0]
        for i in range(iterations):
            p = sigmoid(points * line_parameters)
            gradient = points.T * (p - y) * (alpha / n)
            line_parameters = line_parameters - gradient

            w1 = line_parameters.item(0)
            w2 = line_parameters.item(1)
            b = line_parameters.item(2)

            x1 = np.array([points[:, 0].min(), points[:, 0].max()])
            x2 = -b / w2 + (x1 * (-w1 / w2))
            draw(x1, x2)
            print(p)

        draw(x1.get(),x2.get())


    np.random.seed(0)
    bias = np.ones(n_pts)
    top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, sd1, n_pts), bias]).T
    bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, sd2, n_pts), bias]).T
    all_points = np.vstack((top_region, bottom_region))

    line_parameters = np.matrix([np.zeros(3)]).T

    y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)

    _, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
    ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
    gradient_descent(line_parameters, all_points, y, alpha)
    plt.show()


def call_to():
    global entry1,entry2,entry3,entry4,entry5
    n_pts = int(entry1.get())
    sd1 = int(entry2.get())
    sd2 = int(entry3.get())
    iterations = int(entry5.get())
    alpha = float(entry4.get())
    final_variables(n_pts,sd1,sd2,iterations,alpha)

root = Tk()


root.title('LinearRegressor')
title = Label(root,text = 'LinearRegressor')

label1 = Label(root,text = 'Enter no of points')
label2 = Label(root,text = 'Standard Deviation for label = 1')
label3 = Label(root,text = 'Standard Deviation For label = 2')
label4 = Label(root,text = 'Learning rate')
label5 = Label(root,text = 'Iterations')


entry1 = Entry(root)
entry2 = Entry(root)
entry3 = Entry(root)
entry4 = Entry(root)
entry5 = Entry(root)

button = Button(root,text = 'Press',command = call_to)

label1.grid(row = 0,sticky = E)
label2.grid(row = 1)
label3.grid(row = 2)
label4.grid(row = 3,sticky = E)
label5.grid(row = 4,sticky = E)

entry1.grid(row = 0,column = 1)
#entry1.focus_set()
entry2.grid(row = 1,column = 1)
entry3.grid(row = 2,column = 1)
entry4.grid(row = 3,column = 1)
entry5.grid(row = 4,column = 1)
button.grid(columnspan = 2)
#title.pack()



root.mainloop()

