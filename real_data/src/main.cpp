/*
 * main.cpp
 *
 */

#include "main.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

int main (int argc, char *argv[])
{
	//////////////////////// MPI ///////////////////////////////////
	int MYRANK=0, NP=1;   //DEFAULT assume total one node, current is node 0

#ifdef USE_MPI
	MPI_Status status;  // return status for receive
#endif

#ifdef USE_MPI
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &NP);
	MPI_Comm_rank(MPI_COMM_WORLD, &MYRANK);
#endif

	double dinit = 0.00;
	int i, j, flag, rep;

	int N_0; // samples in class 0
	int N_1; // samples in class 1
	int N; // total number of sample points
	int N_trn;
	int N_tst;
	int D; // total # of features
	int d = 1; // after feature selection
	int max_rep = 1;
	int round = 1;
	long seed = 0L;
	double temp;

	double** X;
	int* y;

	AllErrors all_errors;
	
	double lda_true_error;
	double lsvm_true_error;
	double ksvm_true_error;
	double dlda_true_error;
	double g13_true_error;
	double qda_true_error;
	double sedc_true_error;

	char *dataFilename;
	char *outputFilename;
	char outputFilenameFinal[80];

	FILE *fpIn;

	FILE* fp_lda_true_error;
	FILE* fp_lsvm_true_error;
	FILE* fp_ksvm_true_error;
	FILE* fp_dlda_true_error;
	FILE* fp_g13_true_error;
	FILE* fp_qda_true_error;
	FILE* fp_sedc_true_error;

	//===========================================//
	// Arguments are passed through command-line //
	//===========================================//
	switch (parse_common_arguments(
			&dataFilename,
			&outputFilename,
			&N_trn,
			&d,
			&max_rep,
			&round,
			argc, argv))
	{
	case -1:
	printf("Error in function parse_common_arguments()!\n");
	return(-1);
	case -2:
	printf("Usage printed\n"); // should add some usage printing function here
	return(0);
	default:
		printf("\n");
	}

	fpIn = fopen(dataFilename, "r");
	if(fpIn==NULL)
	{
		printf("cannot open data file in node %d!\n", MYRANK);
		return(-1);
	}

	//obtain the size information of two classes
	fscanf(fpIn,"%d, %d\r\n", &N_0, &N_1);
	
	// total number of features
	fscanf(fpIn,"%d\r\n", &D);

	//size_class1 = size_class0;
	N = N_0 + N_1;
	N_tst = N - N_trn;

	// initializing matrix of original data set
	X = make_2D_matrix(N, D, dinit);
	y = new int [N];

	// obtain the gene data from data file (rows: fetaures, columns: patients)
	// but we transpose it to: (rows: patients, columns: features)
	for (i=0; i<N; i++)
	{
		if (i<N_0)
			y[i] = 0;
		else
			y[i] = 1;
	}

	for (i=0; i<D; i++)
	{
		for (j=0; j<N; j++)
		{
			fscanf(fpIn, "%lf\t", &temp);
			X[j][i] = temp;
		}
		fscanf(fpIn, "\r\n");
	}

	fclose(fpIn);
	
	srand((unsigned)time(NULL)+MYRANK);
	seed = -abs((long)floor(rand()*time(NULL)+MYRANK+1));

	if (MYRANK==0)
	{
		sprintf(outputFilenameFinal, "%s_lda_true_error_round=%d.txt", outputFilename, round);
		fp_lda_true_error = fopen(outputFilenameFinal, "w");
		if(fp_lda_true_error==NULL)
		{
			printf("cannot open %s_lda_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_lsvm_true_error_round=%d.txt", outputFilename, round);
		fp_lsvm_true_error = fopen(outputFilenameFinal, "w");
		if(fp_lsvm_true_error==NULL)
		{
			printf("cannot open %s_lsvm_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_ksvm_true_error_round=%d.txt", outputFilename, round);
		fp_ksvm_true_error = fopen(outputFilenameFinal, "w");
		if(fp_ksvm_true_error==NULL)
		{
			printf("cannot open %s_ksvm_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_dlda_true_error_round=%d.txt", outputFilename, round);
		fp_dlda_true_error = fopen(outputFilenameFinal, "w");
		if(fp_dlda_true_error==NULL)
		{
			printf("cannot open %s_dlda_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_g13_true_error_round=%d.txt", outputFilename, round);
		fp_g13_true_error = fopen(outputFilenameFinal, "w");
		if(fp_g13_true_error==NULL)
		{
			printf("cannot open %s_g13_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_qda_true_error_round=%d.txt", outputFilename, round);
		fp_qda_true_error = fopen(outputFilenameFinal, "w");
		if(fp_qda_true_error==NULL)
		{
			printf("cannot open %s_qda_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
		sprintf(outputFilenameFinal, "%s_sedc_true_error_round=%d.txt", outputFilename, round);
		fp_sedc_true_error = fopen(outputFilenameFinal, "w");
		if(fp_sedc_true_error==NULL)
		{
			printf("cannot open %s_sedc_true_error_round=%d.txt", outputFilename, round);
			return(-1);
		}
	
	}

	for (rep=MYRANK; rep<max_rep; rep+=NP)
	{
		calculateErrors(X,
				y,
				N_trn,
				N_tst,
				D,
				d,
				&seed,
				&all_errors);

		lda_true_error = all_errors.lda_true_error;
		lsvm_true_error = all_errors.lsvm_true_error;
		ksvm_true_error = all_errors.ksvm_true_error;
		dlda_true_error = all_errors.dlda_true_error;
		g13_true_error = all_errors.g13_true_error;
		qda_true_error = all_errors.qda_true_error;
		sedc_true_error = all_errors.sedc_true_error;

		if (MYRANK==0)
		{
			fprintf(fp_lda_true_error, "%f\n", lda_true_error);
			fprintf(fp_lsvm_true_error, "%f\n", lsvm_true_error);
			fprintf(fp_ksvm_true_error, "%f\n", ksvm_true_error);
			fprintf(fp_dlda_true_error, "%f\n", dlda_true_error);
			fprintf(fp_g13_true_error, "%f\n", g13_true_error);
			fprintf(fp_qda_true_error, "%f\n", qda_true_error);
			fprintf(fp_sedc_true_error, "%f\n", sedc_true_error);

#ifdef USE_MPI
			flag = ((max_rep-rep)>(max_rep)%NP ? NP : max_rep%NP);
			for (j=1; j<flag; j++)
			 /*MPI implementation could be here */
#endif
		}
#ifdef USE_MPI
		else
            /*MPI implementation could be here */
#endif
	}

	if (MYRANK==0)
	{
		fclose(fp_lda_true_error);
		fclose(fp_lsvm_true_error);
		fclose(fp_ksvm_true_error);
		fclose(fp_dlda_true_error);
		fclose(fp_g13_true_error);
		fclose(fp_qda_true_error);
		fclose(fp_sedc_true_error);
	}

	delete_2D_matrix(N, D, X);
	delete y;
	
#ifdef USE_MPI
	/*MPI implementation could be here */
#endif

	return 0;
}

