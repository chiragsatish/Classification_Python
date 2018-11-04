import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

f='bank-additional-full.csv'
dataframe = pd.read_csv(f,';')
categoricalFeatures = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
for each in categoricalFeatures:
	col = dataframe[each]
	le = preprocessing.LabelEncoder()
	le.fit(col)
	output=le.transform(col).tolist()
	df = pd.Series(output)
	dataframe[each] = df
df_majority = dataframe[dataframe.y==0]
df_minority = dataframe[dataframe.y==1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=df_majority.shape[0], random_state=123) 
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled = df_upsampled.sample(frac=1)
df_upsampled = df_upsampled.values
print("DECISION TREE BASED CLASSIFICATION\n")
kf = model_selection.KFold(n_splits=5)
accuracysum=precisionsum=recallsum=fmeasuresum=total=0
for train, test in kf.split(df_upsampled):
	train_data = np.array(df_upsampled)[train]
	test_data = np.array(df_upsampled)[test]
	dt = tree.DecisionTreeClassifier()
	train_classificationclass = train_data[:,-1]
	train_data = train_data[:,:-1]
	test_classificationclass = test_data[:,-1]
	test_data = test_data[:,:-1]
	dt = dt.fit(train_data,train_classificationclass)
	predictions = dt.predict(test_data)
	C = metrics.confusion_matrix(test_classificationclass,predictions)
	print("Confusion Matrix:")
	print(C)
	tn, fp, fn, tp = C.ravel()
	accuracy=precision=recall=fmeasure=0.0
	try:
		accuracy = (float(tp)+float(tn))/(float(tp)+float(tn)+float(fp)+float(fn))
		precision = (float(tp))/(float(tp)+float(fp))
		recall = (float(tp))/(float(tp)+float(fn))
		fmeasure = (2*precision*recall)/(precision+recall)
		accuracysum+=accuracy
		precisionsum+=precision
		recallsum+=recall
		fmeasuresum+=fmeasure
		total+=1
	except:
		print("Divide by Zero Error")
	print("Accuracy:%f \nPrecision:%f \nRecall:%f \nF-Measure:%f \n" % (accuracy,precision,recall,fmeasure))
accuracysum = accuracysum/total
precisionsum=precisionsum/total
recallsum=recallsum/total
fmeasuresum=fmeasuresum/total
print("Avg. Accuracy:%f \nAvg. Precision:%f \nAvg. Recall:%f \nAvg. F-Measure:%f \n" % (accuracysum,precisionsum,recallsum,fmeasuresum))

print("NEURAL NETWORK BASED CLASSIFICATION")
kf = model_selection.KFold(n_splits=5)
accuracysum=precisionsum=recallsum=fmeasuresum=total=0
for train, test in kf.split(df_upsampled):
	train_data = np.array(df_upsampled)[train]
	test_data = np.array(df_upsampled)[test]
	mlp = MLPClassifier()
	train_classificationclass = train_data[:,-1]
	train_data = train_data[:,:-1]
	test_classificationclass = test_data[:,-1]
	test_data = test_data[:,:-1]
	mlp = mlp.fit(train_data,train_classificationclass)
	predictions = mlp.predict(test_data)
	C = metrics.confusion_matrix(test_classificationclass,predictions)
	print("Confusion Matrix:")
	print(C)
	tn, fp, fn, tp = C.ravel()
	accuracy=precision=recall=fmeasure=0.0
	try:
		accuracy = (float(tp)+float(tn))/(float(tp)+float(tn)+float(fp)+float(fn))
		precision = (float(tp))/(float(tp)+float(fp))
		recall = (float(tp))/(float(tp)+float(fn))
		fmeasure = (2*precision*recall)/(precision+recall)
		accuracysum+=accuracy
		precisionsum+=precision
		recallsum+=recall
		fmeasuresum+=fmeasure
		total+=1
	except:
		print("Divide by Zero Error")
	print("Accuracy:%f \nPrecision:%f \nRecall:%f \nF-Measure:%f \n" % (accuracy,precision,recall,fmeasure))
accuracysum = accuracysum/total
precisionsum=precisionsum/total
recallsum=recallsum/total
fmeasuresum=fmeasuresum/total
print("Avg. Accuracy:%f \nAvg. Precision:%f \nAvg. Recall:%f \nAvg. F-Measure:%f \n" % (accuracysum,precisionsum,recallsum,fmeasuresum))
print("*******NOTE*******")
print("As we get 0's in a few entries of the confusion matrix, sometimes we are not able to determine all 4 metrics in all folds.\nHence, only folds where all the metrics can be calculated have been considered.")
