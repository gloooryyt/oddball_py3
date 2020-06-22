'''
Python3 implementation of oddball

@author:
Tao Yu (gloooryyt@gmail.com)

'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

# feature dictionary which format is {node i's id:Ni, Ei, Wi, λw,i}

def star_or_clique(featureDict):
    N = []
    E = []
    for key in featureDict.keys():
        N.append(featureDict[key][0])
        E.append(featureDict[key][1])
    #E=CN^α => log on both sides => logE=logC+αlogN
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.log2(E)
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(E), 1)
    x_train = np.log2(N)
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(N), 1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    alpha = w
    outlineScoreDict = {}
    for key in featureDict.keys():
        yi = featureDict[key][1]
        xi = featureDict[key][0]
        outlineScore = (max(yi, C*(xi**alpha))/min(yi, C*(xi**alpha)))*np.log(abs(yi-C*(xi**alpha))+1)
        outlineScoreDict[key] = outlineScore
    return outlineScoreDict


def heavy_vicinity(featureDict):
    W = []
    E = []
    for key in featureDict.keys():
        W.append(featureDict[key][2])
        E.append(featureDict[key][1])
    #W=CE^β => log on both sides => logW=logC+βlogE
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.log2(W)
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(W), 1)
    x_train = np.log2(E)
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(E), 1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    beta = w
    outlineScoreDict = {}
    for key in featureDict.keys():
        yi = featureDict[key][2]
        xi = featureDict[key][1]
        outlineScore = (max(yi, C*(xi**beta))/min(yi, C*(xi**beta)))*np.log(abs(yi-C*(xi**beta))+1)
        outlineScoreDict[key] = outlineScore
    return outlineScoreDict


def dominant_edge(featureDict):
    Lambda_w_i = []
    W = []
    for key in featureDict.keys():
        Lambda_w_i.append(featureDict[key][3])
        W.append(featureDict[key][2])
    #λ=CW^γ => log on both sides => logλ=logC+γlogW
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.log2(Lambda_w_i)
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(Lambda_w_i), 1)
    x_train = np.log2(W)
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(W), 1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2 ** b
    beta = w
    outlineScoreDict = {}
    for key in featureDict.keys():
        yi = featureDict[key][3]
        xi = featureDict[key][2]
        outlineScore = (max(yi, C * (xi ** beta)) / min(yi, C * (xi ** beta))) * np.log(abs(yi - C * (xi ** beta)) + 1)
        outlineScoreDict[key] = outlineScore
    return outlineScoreDict


def star_or_clique_withLOF(featureDict):
    N = []
    E = []
    for key in featureDict.keys():
        N.append(featureDict[key][0])
        E.append(featureDict[key][1])
    #E=CN^α => log on both sides => logE=logC+αlogN
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.log2(E)
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(E), 1)
    x_train = np.log2(N)
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(N), 1)    #the order in x_train and y_train is the same as which in featureDict.keys() now

    #prepare data for LOF
    xAndyForLOF = []
    for index in range(len(N)):
        tempArray = np.array([x_train[index][0], y_train[index][0]])
        xAndyForLOF.append(tempArray)
    xAndyForLOF = np.array(xAndyForLOF)

    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    alpha = w
    print('alpha={}'.format(alpha))

    #LOF algorithm
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit(xAndyForLOF)
    LOFScoreArray = -clf.negative_outlier_factor_

    outScoreDict = {}
    count = 0   #Used to take LOFScore in sequence from LOFScoreArray

    #get the maximum outLine
    maxOutLine = 0
    for key in featureDict.keys():
        yi = featureDict[key][1]
        xi = featureDict[key][0]
        outlineScore = (max(yi, C*(xi**alpha))/min(yi, C*(xi**alpha)))*np.log(abs(yi-C*(xi**alpha))+1)
        if outlineScore > maxOutLine:
            maxOutLine = outlineScore

    print('maxOutLine={}'.format(maxOutLine))

    #get the maximum LOFScore
    maxLOFScore = 0
    for ite in range(len(N)):
        if LOFScoreArray[ite] > maxLOFScore:
            maxLOFScore = LOFScoreArray[ite]

    print('maxLOFScore={}'.format(maxLOFScore))

    for key in featureDict.keys():
        yi = featureDict[key][1]
        xi = featureDict[key][0]
        outlineScore = (max(yi, C*(xi**alpha))/min(yi, C*(xi**alpha)))*np.log(abs(yi-C*(xi**alpha))+1)
        LOFScore = LOFScoreArray[count]
        count += 1
        outScore = outlineScore/maxOutLine + LOFScore/maxLOFScore
        outScoreDict[key] = outScore
    return outScoreDict


def heavy_vicinity_withLOF(featureDict):
    W = []
    E = []
    for key in featureDict.keys():
        W.append(featureDict[key][2])
        E.append(featureDict[key][1])
    #W=CE^β => log on both sides => logW=logC+βlogE
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.log2(W)
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(W), 1)
    x_train = np.log2(E)
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(E), 1)    #the order in x_train and y_train is the same as which in featureDict.keys() now

    #prepare data for LOF
    xAndyForLOF = []
    for index in range(len(W)):
        tempArray = np.array([x_train[index][0], y_train[index][0]])
        xAndyForLOF.append(tempArray)
    xAndyForLOF = np.array(xAndyForLOF)

    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    beta = w
    print('beta={}'.format(beta))

    #LOF algorithm
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit(xAndyForLOF)
    LOFScoreArray = -clf.negative_outlier_factor_

    outScoreDict = {}
    count = 0   #Used to take LOFScore in sequence from LOFScoreArray

    #get the maximum outLine
    maxOutLine = 0
    for key in featureDict.keys():
        yi = featureDict[key][2]
        xi = featureDict[key][1]
        outlineScore = (max(yi, C*(xi**beta))/min(yi, C*(xi**beta)))*np.log(abs(yi-C*(xi**beta))+1)
        if outlineScore > maxOutLine:
            maxOutLine = outlineScore

    print('maxOutLine={}'.format(maxOutLine))

    #get the maximum LOFScore
    maxLOFScore = 0
    for ite in range(len(W)):
        if LOFScoreArray[ite] > maxLOFScore:
            maxLOFScore = LOFScoreArray[ite]

    print('maxLOFScore={}'.format(maxLOFScore))

    for key in featureDict.keys():
        yi = featureDict[key][2]
        xi = featureDict[key][1]
        outlineScore = (max(yi, C*(xi**beta))/min(yi, C*(xi**beta)))*np.log(abs(yi-C*(xi**beta))+1)
        LOFScore = LOFScoreArray[count]
        count += 1
        outScore = outlineScore/maxOutLine + LOFScore/maxLOFScore
        outScoreDict[key] = outScore
    return outScoreDict

def dominant_edge_withLOF(featureDict):
    Lambda_w_i = []
    W = []
    for key in featureDict.keys():
        Lambda_w_i.append(featureDict[key][3])
        W.append(featureDict[key][2])
    #λ=CW^γ => log on both sides => logλ=logC+γlogW
    #regard as y=b+wx to do linear regression
    #here the base of log is 2
    y_train = np.log2(Lambda_w_i)
    y_train = np.array(y_train)
    y_train = y_train.reshape(len(Lambda_w_i), 1)
    x_train = np.log2(W)
    x_train = np.array(x_train)
    x_train = x_train.reshape(len(W), 1)    #the order in x_train and y_train is the same as which in featureDict.keys() now

    #prepare data for LOF
    xAndyForLOF = []
    for index in range(len(W)):
        tempArray = np.array([x_train[index][0], y_train[index][0]])
        xAndyForLOF.append(tempArray)
    xAndyForLOF = np.array(xAndyForLOF)

    model = LinearRegression()
    model.fit(x_train, y_train)
    w = model.coef_[0][0]
    b = model.intercept_[0]
    C = 2**b
    gamma = w
    print('gamma={}'.format(gamma))

    #LOF algorithm
    clf = LocalOutlierFactor(n_neighbors=20)
    clf.fit(xAndyForLOF)
    LOFScoreArray = -clf.negative_outlier_factor_

    outScoreDict = {}
    count = 0   #Used to take LOFScore in sequence from LOFScoreArray

    #get the maximum outLine
    maxOutLine = 0
    for key in featureDict.keys():
        yi = featureDict[key][3]
        xi = featureDict[key][2]
        outlineScore = (max(yi, C*(xi**gamma))/min(yi, C*(xi**gamma)))*np.log(abs(yi-C*(xi**gamma))+1)
        if outlineScore > maxOutLine:
            maxOutLine = outlineScore

    print('maxOutLine={}'.format(maxOutLine))

    #get the maximum LOFScore
    maxLOFScore = 0
    for ite in range(len(W)):
        if LOFScoreArray[ite] > maxLOFScore:
            maxLOFScore = LOFScoreArray[ite]

    print('maxLOFScore={}'.format(maxLOFScore))

    for key in featureDict.keys():
        yi = featureDict[key][3]
        xi = featureDict[key][2]
        outlineScore = (max(yi, C*(xi**gamma))/min(yi, C*(xi**gamma)))*np.log(abs(yi-C*(xi**gamma))+1)
        LOFScore = LOFScoreArray[count]
        count += 1
        outScore = outlineScore/maxOutLine + LOFScore/maxLOFScore
        outScoreDict[key] = outScore
    return outScoreDict
