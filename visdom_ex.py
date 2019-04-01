#coding: utf-8

import numpy as np 
import visdom
import time

vis = visdom.Visdom()
win = vis.line(
    X = np.column_stack((np.array([0]), np.array([0]))),
    Y = np.column_stack((np.array([0]), np.array([0]))),
)

for x in np.arange(120):
    y1 = np.random.randn(1)*5
    y2 = np.random.randn(1)*3+10
    vis.line(
        X = np.column_stack((np.array([x]), np.array([x]))),
        Y = np.column_stack((np.array([y1]), np.array([y2]))), 
        win = win, 
        update = 'append'
    )
    time.sleep(1)
