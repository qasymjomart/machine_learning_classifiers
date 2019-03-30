/*
 * calculateErrors.h
 *
 */

#ifndef CALCULATEERRORS_H_
#define CALCULATEERRORS_H_

#include "standardHeaders.h"
#include "print.h"
#include "utilities.h"
#include "matrixOperations.h"
#include "dataGeneration.h"
#include "featureSelection.h"
#include "classifiers.h"


struct AllErrors {
	double lda_true_error;
	double lsvm_true_error;
	double ksvm_true_error;
	double dlda_true_error;
	double g13_true_error;
	double qda_true_error;
	double sedc_true_error;
	
};

void calculateErrors(
		double** X,
		int* y,
		int N_trn,
		int N_tst,
		int D,
		int d,
		long* seed,
		AllErrors* all_errors);

#endif /* CALCULATEERRORS_H_ */
