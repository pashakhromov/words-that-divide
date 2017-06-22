#!/usr/bin/env python

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np

if __name__ == '__main__':
	newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])

	X = newsgroups.data
	y = newsgroups.target
	
	seed = 241
	
	
	# Pre-processing
	vectorizer = TfidfVectorizer()
	X_train = vectorizer.fit_transform(X)
	X_test = vectorizer.transform(X)
	
	
	# Find best regularization
	grid = {'C': np.power(10.0, np.arange(-5, 6))}
	cv = KFold(n_splits=5, shuffle=True, random_state=seed)
	clf = SVC(kernel='linear', random_state=seed)
	gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
	gs.fit(X_train, y)

	print 'summary\n', gs.cv_results_
	print 'best param\n', gs.best_params_
	print 'best score\n', gs.best_score_
	
	
	# With best model, find top 10 words that are the strongest predictors
	best_clf = SVC(C=gs.best_params_['C'], kernel='linear', random_state=seed)
	best_clf.fit(X_train, y)
	coeffs = best_clf.coef_
	print coeffs.shape, coeffs.shape[1]

	c = np.zeros(coeffs.shape[1])
	for i in range(coeffs.shape[1]):
		if (coeffs[0,i] < 0):
			c[i] = -coeffs[0,i]
		else:
			c[i] = coeffs[0,i]

	ind = np.argsort(c)
	feature_mapping = vectorizer.get_feature_names()

	print 'top 10 words'
	words = []
	for i in ind[-10:]:
		#print feature_mapping[i], coeffs[0,i]
		words.append(feature_mapping[i])
	print words
