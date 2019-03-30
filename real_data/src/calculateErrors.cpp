/*
 * calculateErrors.cpp
 *
 *  Created on: Apr 3, 2012
 *      Author: General
 */

#include "calculateErrors.h"

void calculateErrors(
		double** X,
		int* y,
		int N_trn,
		int N_tst,
		int D,
		int d,
		long* seed,
		AllErrors* all_errors)
{
	double dinit = 0.00;

	SimulationData data_trn;
	SimulationData data_tst;

	int* best_features;

	LDA model_LDA;
	DLDA model_DLDA;
	G13 model_G13;
	QDA model_QDA;
	SEDC model_SEDC;

	svm_model *model_LSVM;	//svm training model
	svm_node *subdata_LSVM;	//svm training data
	svm_problem subcl_LSVM;	//svm training data structure

	svm_model *model_KSVM;	//svm training model
	svm_node *subdata_KSVM;	//svm training data
	svm_problem subcl_KSVM;	//svm training data structure

	(*all_errors).lda_true_error = 0.00;
	(*all_errors).lsvm_true_error = 0.00;
	(*all_errors).ksvm_true_error = 0.00;
	(*all_errors).dlda_true_error = 0.00;
	(*all_errors).g13_true_error = 0.00;
	(*all_errors).qda_true_error = 0.00;
	(*all_errors).sedc_true_error = 0.00;

	data_trn.data = make_2D_matrix(N_trn, D, dinit);
	data_trn.labels = new int [N_trn];
	data_tst.data = make_2D_matrix(N_tst, D, dinit);
	data_tst.labels = new int [N_tst];

	dataGeneration(X, y, N_trn, N_tst, D, seed, &data_trn, &data_tst);

	best_features = new int [d];
	featureSelection(data_trn.data, data_trn.labels, data_trn.N, data_trn.D, d, best_features);
	
	model_LDA.a = new double [d];
	if(d < N_trn)
	ldaTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, &model_LDA);
	else ;
	
	model_DLDA.da = new double [d];
	dldaTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, &model_DLDA);
	
	model_G13.ga = new double [d];
	if(d < N_trn) g13Trn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, &model_G13);
	else ;
	
	model_QDA.qa = new double [d];
	model_QDA.mean_0 = new double [d];
	model_QDA.mean_1 =	new double [d];
	model_QDA.inv_cov_0 = make_2D_matrix(d, d, dinit);
	model_QDA.inv_cov_1 =	make_2D_matrix(d, d, dinit);
	qdaTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, &model_QDA);
	
	int m = 5;
	int k = d/m;
	model_SEDC.z_temp_1 = new double [k];
	model_SEDC.z_temp_2 = new double [k];
	sedcTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, &model_SEDC);

	model_LSVM = svmTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, 0, &subdata_LSVM, &subcl_LSVM);
	model_KSVM = svmTrn(data_trn.data, data_trn.labels, data_trn.N, d, best_features, 2, &subdata_KSVM, &subcl_KSVM);

	if(d < N_trn) {
	(*all_errors).lda_true_error = ldaTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_LDA);
	(*all_errors).g13_true_error = g13Tst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_G13);
	} else {
	(*all_errors).lda_true_error = 0.00;
	(*all_errors).g13_true_error = 0.00;	
	}
	(*all_errors).dlda_true_error = dldaTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_DLDA);
	(*all_errors).lsvm_true_error = svmTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_LSVM);
	(*all_errors).ksvm_true_error = svmTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_KSVM);
	(*all_errors).qda_true_error = qdaTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_QDA);
	(*all_errors).sedc_true_error = sedcTst(data_tst.data, data_tst.labels, data_tst.N, d, best_features, model_SEDC);

	
	delete model_LDA.a;
	delete model_DLDA.da;
	delete model_G13.ga;
	delete model_QDA.qa;
	delete model_QDA.mean_0;
	delete model_QDA.mean_1;
	delete_2D_matrix(d, d, model_QDA.inv_cov_0);
	delete_2D_matrix(d, d, model_QDA.inv_cov_1);
	delete best_features;

	svmDestroy(model_LSVM, subdata_LSVM, &subcl_LSVM);
	svmDestroy(model_KSVM, subdata_KSVM, &subcl_KSVM);

	delete_2D_matrix(N_trn, D, data_trn.data);
	delete data_trn.labels;
	delete_2D_matrix(N_tst, D, data_tst.data);
	delete data_tst.labels;

	return;
}
