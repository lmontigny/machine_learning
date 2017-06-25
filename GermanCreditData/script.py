# German Credit Data 
# 2016-12-01
# Created by Laurent Montigny

import os
import shutil
import IPython
import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics  #sklearn logistic regression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel #feature selection
from sklearn.feature_selection import RFE #Recursive Feature Elimination
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import ExtraTreesClassifier #feature selection
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split 
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.learning_curve import learning_curve
from pandas.tools.plotting import scatter_matrix
#from sklearn.model_selection import learning_curve #sklearn 0.18

# ------------------------------ #
#             Function()         #
# ------------------------------ #

def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

J_history=[]
def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))
    J_history.append(J)
               
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    grad =(1/m)*X.T.dot(h-y)
    return(grad.flatten())

def calculateStatistics(df, X):
    stat_1 = df.describe()
    stat_2 = df.std()

    # Box plot
    a=df.select_dtypes(include=['int64'])
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
    axes[0].boxplot(a.Duration_Months) 
    axes[0].set_title('Duration_Months')

    axes[1].boxplot(a.amount) 
    axes[1].set_title('amount')

    axes[2].boxplot(a.age) 
    axes[2].set_title('age')

    fig.subplots_adjust(hspace=0.4)
    fig.savefig("plot/box_plot.pdf", bbox_inches='tight')
    plt.close(fig)
    return stat_1

def read_txt():
    df = pd.read_csv('german.data', sep=" ", header = None)
    df.columns = ["check_Acc_Status" , "Duration_Months" , "Credit_history" , "Credit_purpose" ,       
            "amount" , "savings" , "employ_since" , "installment_rate" , "status_sex" ,        
            "cosigners" , "residence_since", "collateral" , "age" , "otherplans" ,   
            "housing" , "existing_credits" , "job" , "no_dependents" , "telephone",           
            "foreign", "default"]
    return df

def arrange_data(df):
    # Categorize the data (object->int)  #Only for plot
    #df0 = df.copy()
    #for column in df0:
        #if df0[column].dtype == np.object:
            #df0[column] = df0[column].astype('category')
            #df0[column] = df0[column].cat.codes 
    #fig, ax = plt.subplots(figsize=(20, 12))
    #df0.hist(ax=ax)
    #fig.savefig("plot/data_frame.pdf", dpi=200, bbox_inches='tight')
    #plt.close()


    for column in df:
        # Create dummy variables (0,1) for categorical variable
        if df[column].dtype == np.object:
            dummies = pd.get_dummies(df[column]).rename(columns=lambda x: column +'_' + str(x))
            df = pd.concat([df, dummies], axis=1)
            df = df.drop([column], axis=1)

    # Normalize quantitative variable
    scaler = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df),columns=df.columns)

    y = df_scaled.default
    X = df_scaled.drop(['default'],axis=1)
    X_df = X.copy()
    return df_scaled, X_df ,X.as_matrix(), y.as_matrix()

def feature_selection(model, X,y):
    print("Model for feature selection used: %i" % model)
    if (model==1):
        model = ExtraTreesClassifier()
        features = feature_selection_forest(model,X,y)
        return features
    elif (model == 2):
        model = RandomForestRegressor()
        features = feature_selection_forest(model,X,y)
        return features
    elif (model == 3):
        model = LogisticRegression()
        features = feature_selection_RFE(model,X,y)
        return features

def feature_selection_forest(model,X,y):
    fit = model.fit(X,y.ravel())
    toto = SelectFromModel(fit, prefit=True)
    features = toto.transform(X)
    # features = fit.transform(X)

    importances = fit.feature_importances_ 
    std = np.std([tree.feature_importances_ for tree in fit.estimators_],
                         axis=0)
    indices = np.argsort(importances)[::-1]

    # Plot importance
    fig = plt.figure(dpi=600)
    X_name = list(X.columns.values) #test
    feature_importance = fit.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)[::-1]
    # print "Feature importance:"
    # i=1
    # for f,w in zip(X.columns[sorted_idx], feature_importance[sorted_idx]):
    #         print "%d) %s : %d" % (i, f, w)
    #         i+=1
    pos = np.arange(sorted_idx.shape[0]) + .5
    nb_to_display = features.shape[1]
    plt.barh(pos[:nb_to_display], feature_importance[sorted_idx][:nb_to_display], color='b',  align='center')
    plt.yticks(pos[:nb_to_display], X.columns[sorted_idx][:nb_to_display])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.legend(('Tree','Forest'))
    # plt.show()
    fig.savefig("plot/feature_selection.pdf", bbox_inches='tight')
    plt.close(fig)

    print("Number of features: %d" %features.shape[1])
    return features 


def feature_selection_RFE(model,X,y):
    rfe = RFE(model, 20)
    fit = rfe.fit(X, y.ravel())
    features = fit.transform(X)
    print("Number of features: %d") % fit.n_features_
    # print("Selected Features: %s") % fit.support_
    # print("Feature Ranking: %s") % fit.ranking_
    return features 

def plot_cost_function():
    fig = plt.figure()
    plt.xlabel("Iteration")
    plt.ylabel("Cost function J")
    plt.plot(J_history)
    #plt.show()
    fig.savefig("plot/cost_function_logreg.pdf", bbox_inches='tight')
    plt.close()

def log_regression_lrlearn(X_train,y_train,X_test,y_test):
    # Initialize the coefficient vector 
    n = np.shape(X_train)[1]
    #theta0 = np.random.rand(n)*0.01
    #theta0 = np.random.rand(n,1)*0.01
    theta0 = np.zeros(n)

    # Minimize the linear regression objective function
    y_train  = np.c_[y_train]
    cost = costFunction(theta0, X_train, y_train)
    grad = gradient(theta0, X_train, y_train)

    result = opt.minimize(costFunction, theta0, args=(X_train,y_train), method=None, jac=gradient, 
            options={'maxiter':400, 'disp': True})
    #result = opt.minimize(costFunction, theta0, args=(X_train,y_train), method='TNC', tol=1e-10)
    #result = opt.fmin_bfgs(costFunction, theta0, args=(X_train,y_train))
    theta = result.x

    plot_cost_function()

    # Check accuracy
    proba = predict(theta,X_test)
    accuracy = y_test[np.where(proba == y_test)].size / float(y_test.size) * 100.0
    print ('Train Accuracy: %f' % accuracy)
    return proba

def sklearn(model,X_train,y_train,X_test,y_test):
    # fit a model to the data
    model.fit(X_train,y_train.ravel())
    print(model)

    # make predictions
    expected = y_test
    proba = model.predict(X_test)

    # summarize the fit of the model
    print(metrics.classification_report(expected, proba))
    #print(metrics.confusion_matrix(expected, proba))
    accuracy = y_test[np.where(proba == y_test)].size / float(y_test.size) * 100.0
    print ('Train Accuracy: %f' % accuracy)
    return proba

def err_model(proba1, proba2):
    err = (proba1 != proba2).sum()/float(proba1.size) * 100.0
    print('Error with the logistic models: %f %%' % err)
    return err

def cross_validation(model,X,y):
    print('\n Cross validation cross_val_score')
    accuracy = cross_val_score(model,X,y.ravel(), cv=10)
    print(accuracy)
    print("Average accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std() * 2))
    return accuracy.mean()

def cross_validation_KFold(X,y):
    print('\n Cross validation KFold')
    cv = KFold(n=len(X),  n_folds=10)
    accuracy = []

    for train_cv, valid_cv in cv:
        model = LogisticRegression()
        model.fit(X[train_cv],y[train_cv])

        valid_acc = model.score(X[valid_cv],y[valid_cv])
        accuracy.append(valid_acc)    
    print(["%0.2f" % i for i in accuracy])
    print("Average accuracy: %0.2f (+/- %0.2f)" % (np.mean(accuracy), np.std(accuracy) * 2))
    return np.mean(accuracy)


def calc_confusion_matrix(y_true, y_proba):
    conmat0 = confusion_matrix(y_true, y_proba)
    #print("Confusion matrix:")
    #print("Good | Bad Credit")
    #print(conmat0)

    print("\n Confusion matrix:")
    conmat=pd.crosstab(y_true, y_proba, rownames=['True'], colnames=['Predicted'], margins=True)
    print(conmat)

    #Plot Confusion matrix
    labels = ['Good', 'Bad']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conmat0)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fig.savefig("plot/confusion_matrix.pdf", bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    
    return conmat0

def plot_learning_curve(model, title, X, y):
    #provided by scikit-learn.
    fig = plt.figure()
    plt.title(title)
    #if ylim is not None:
        #plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    #train_sizes=np.linspace(.1, 1.0, 5)
    #train_sizes, train_scores, test_scores = learning_curve( model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    fig.savefig("plot/learning_curve_"+title+".pdf", bbox_inches='tight')
    plt.close(fig)

def plot_scatter_matrix(df):
    plt.close("all")
    scatter_matrix(df, alpha=0.2, figsize=(20, 20), diagonal='kde')
    plt.savefig("plot/scatter_matrix.pdf", dpi=100, bbox_inches='tight')
    plt.close()

def plot_crosstab(df,df_component):
    a=pd.crosstab(df.default,df_component,rownames=['default'])
    a.plot(kind='bar')
    plt.savefig("plot/cross_tab.pdf", dpi=100, bbox_inches='tight')
    plt.close()

def plot_accuracy(A):
    fig, ax = plt.subplots()
    plt.bar(np.arange(A.size),A)
    ax.set_xticklabels([''] + model_list, rotation='vertical')
    plt.subplots_adjust(bottom=0.15)
    #adjust x ticks
    for axis in [ax.xaxis]:
        axis.set(ticks=np.arange(0.5, len(model_list)), ticklabels=model_list)
    fig.savefig("plot/accuracy.pdf", bbox_inches='tight')
    plt.close(fig)

def clean_folder():
    folder = "plot/"
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=False, onerror=None) 
    os.mkdir(folder)

# ------------------------------ #
#             Main()             #
# ------------------------------ #

# def main():

# Create plot folder
#clean_folder()

# Get data
df1 = read_txt()

# Tranform data
df, X_df, X, y = arrange_data(df1)
#X = np.c_[np.ones((X.shape[0],1)), X]  #need it?

# Statistics, exploration-phase
stat = calculateStatistics(df1, X)
plot_crosstab(df, df.Credit_purpose_A40)
# plot_scatter_matrix(df1.drop(['default'],axis=1)) # slow

# Feature selection
model_feature = 1  #{1=ExtraTreesClassifier(),2=RandomForestRegressor(),3=LogisticRegression()}
X_reduced = feature_selection(model_feature, X_df,y)

# Separate data
X_train, X_test, y_train, y_test = train_test_split(X_reduced,y,test_size=0.2) 

# Model prediction (LogisticRegressionLR is implemented step by step)
err_matrix=[]
proba_matrix=[]
accuracy_matrix=[]
confusion_global=np.array([])

model_list = ["LogisticRegressionLR", "LogisticRegression()", "SVC()", 
        "DecisionTreeClassifier()", "RandomForestClassifier()", "GaussianNB()"]
counter = 0
for i in model_list:
    if (i == "LogisticRegressionLR"):
        proba_matrix.append(log_regression_lrlearn(X_train,y_train,X_test,y_test))
        accuracy = cross_validation_KFold(X_reduced,y) #test with KFold
        accuracy_matrix.append(accuracy)
        conmat = calc_confusion_matrix(y_test, proba_matrix[0])
        confusion_global = np.append(confusion_global,conmat)
    else:
        model=eval(model_list[counter])
        print("Model sklearn used: %s \n" %model_list[counter])

        # Make prediction with the model
        proba = sklearn(model,X_train,y_train,X_test,y_test)
        proba_matrix.append(proba)

        # Cross validation
        #cross_validation_KFold(X_reduced,y) #test with KFold
        accuracy = cross_validation(model,X_reduced,y)
        accuracy_matrix.append(accuracy)

        # Plot learning curve
        plot_learning_curve(model, model_list[counter], X_reduced,y)

    # Error with the log_regression method
    err = err_model(proba_matrix[0], proba_matrix[counter])
    err_matrix.append(err)

    # Confusion Matrix
    conmat = calc_confusion_matrix(y_test, proba_matrix[counter])
    confusion_global = np.append(confusion_global,conmat)

    counter = counter + 1

E = np.asarray(err_matrix.copy())
P = np.asarray(proba_matrix.copy())
A = np.asarray(accuracy_matrix.copy())
C = np.reshape(confusion_global,(-1,2))

plot_accuracy(A)


#IPython.embed()

# if __name__ == '__main__':
#     main()
