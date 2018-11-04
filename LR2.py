import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

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
dataframe = dataframe.values
kf = model_selection.KFold(n_splits=5)
accuracysum=precisionsum=recallsum=fmeasuresum=0
for train, test in kf.split(dataframe):
	train_data = np.array(dataframe)[train]
	test_data = np.array(dataframe)[test]
	lr = linear_model.LogisticRegression()
	train_classificationclass = train_data[:,-1]
	train_data = train_data[:,:-1]
	test_classificationclass = test_data[:,-1]
	test_data = test_data[:,:-1]
	lr.fit(train_data,train_classificationclass)
	predictions = lr.predict(test_data)
	C = metrics.confusion_matrix(test_classificationclass,predictions)
	print("Confusion Matrix:")
	print(C)
	tn, fp, fn, tp = C.ravel()
	accuracy = (float(tp)+float(tn))/(float(tp)+float(tn)+float(fp)+float(fn))
	precision = (float(tp))/(float(tp)+float(fp))
	recall = (float(tp))/(float(tp)+float(fn))
	fmeasure = (2*precision*recall)/(precision+recall)
	print("Accuracy:%f \nPrecision:%f \nRecall:%f \nF-Measure:%f \n" % (accuracy,precision,recall,fmeasure))
	accuracysum+=accuracy
	precisionsum+=precision
	recallsum+=recall
	fmeasuresum+=fmeasure
accuracysum = accuracysum/5
precisionsum=precisionsum/5
recallsum=recallsum/5
fmeasuresum=fmeasuresum/5
print("Avg. Accuracy:%f \nAvg. Precision:%f \nAvg. Recall:%f \nAvg. F-Measure:%f \n" % (accuracysum,precisionsum,recallsum,fmeasuresum))

