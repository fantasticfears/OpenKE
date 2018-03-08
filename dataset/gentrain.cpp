#include <Python.h>
#include <cstdlib>
#include <algorithm>
#include "Random.h"
#include "Triple.h"

long long num_triplets;
long long read_triplets;
long long num_entities;
long long num_relations;

int *freqRel, *freqEnt;
int *lefHead, *rigHead;
int *lefTail, *rigTail;
int *lefRel, *rigRel;
double *left_mean, *right_mean;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;

#include "Corrupt.h"

static PyObject *
gentrain_init_buff(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, "LLL", &num_triplets, &num_entities, &num_relations)) {
        return NULL;
	} else {
		read_triplets = 0;
		trainList = static_cast<Triple *>(calloc(num_triplets, sizeof(Triple)));
		trainHead = static_cast<Triple *>(calloc(num_triplets, sizeof(Triple)));
		trainTail = static_cast<Triple *>(calloc(num_triplets, sizeof(Triple)));
		trainRel = static_cast<Triple *>(calloc(num_triplets, sizeof(Triple)));
		freqRel = static_cast<int *>(calloc(num_relations, sizeof(int)));
		freqEnt = static_cast<int *>(calloc(num_entities, sizeof(int)));
		lefHead = static_cast<int *>(calloc(num_entities, sizeof(int)));
		rigHead = static_cast<int *>(calloc(num_entities, sizeof(int)));
		lefTail = static_cast<int *>(calloc(num_entities, sizeof(int)));
		rigTail = static_cast<int *>(calloc(num_entities, sizeof(int)));
		lefRel = static_cast<int *>(calloc(num_relations, sizeof(int)));
		rigRel = static_cast<int *>(calloc(num_relations, sizeof(int)));
		left_mean = static_cast<double *>(calloc(num_relations, sizeof(double)));
		right_mean = static_cast<double *>(calloc(num_relations, sizeof(double)));
    	return PyLong_FromLongLong(num_triplets);
	}
	randReset();
}

static PyObject *
gentrain_feed(PyObject *self, PyObject *args)
{
	int h, r, t;
    if (!PyArg_ParseTuple(args, "(iii)", &h, &r, &t))
        return NULL;

	trainList[read_triplets].h = trainHead[read_triplets].h = trainTail[read_triplets].h = h;
	trainList[read_triplets].r = trainHead[read_triplets].r = trainTail[read_triplets].r = r;
	trainList[read_triplets].t = trainHead[read_triplets].t = trainTail[read_triplets].t = t;
	read_triplets++;
    return PyLong_FromLongLong(read_triplets);
}

static PyObject *
gentrain_freq(PyObject *self, PyObject *args)
{
    if (!PyArg_ParseTuple(args, ""))
        return NULL;

	std::sort(trainList, trainList + num_triplets, Triple::cmp_head);
	long long unique_num_triplets = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	freqEnt[trainList[0].t] += 1;
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
	for (int i = 1; i < num_triplets; i++) {
		if (!(trainList[i].h == trainList[i-1].h && trainList[i].r == trainList[i-1].r && trainList[i].t == trainList[i-1].t)) {
			continue;
		}
		trainHead[unique_num_triplets] = trainTail[unique_num_triplets] =
			trainRel[unique_num_triplets] = trainList[unique_num_triplets] = trainList[i];
		unique_num_triplets++;
		freqEnt[trainList[i].t]++;
		freqEnt[trainList[i].h]++;
		freqRel[trainList[i].r]++;
	}

	std::swap(num_triplets, unique_num_triplets);
	std::sort(trainHead, trainHead + num_triplets, Triple::cmp_head);
	std::sort(trainTail, trainTail + num_triplets, Triple::cmp_tail);
	std::sort(trainRel, trainRel + num_triplets, Triple::cmp_rel);

	lefHead = static_cast<int *>(calloc(num_entities, sizeof(int)));
	rigHead = static_cast<int *>(calloc(num_entities, sizeof(int)));
	lefTail = static_cast<int *>(calloc(num_entities, sizeof(int)));
	rigTail = static_cast<int *>(calloc(num_entities, sizeof(int)));
	lefRel = static_cast<int *>(calloc(num_entities, sizeof(int)));
	rigRel = static_cast<int *>(calloc(num_entities, sizeof(int)));
	memset(rigHead, -1, sizeof(int)*num_entities);
	memset(rigTail, -1, sizeof(int)*num_entities);
	memset(rigRel, -1, sizeof(int)*num_entities);
	for (int i = 1; i < num_entities; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h) {
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[num_entities - 1].h] = num_entities - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[num_entities - 1].t] = num_entities - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[num_entities - 1].h] = num_entities - 1;

	left_mean = static_cast<double *>(calloc(num_relations,sizeof(double)));
	right_mean = static_cast<double *>(calloc(num_relations,sizeof(double)));
	for (int i = 0; i < num_entities; i++) {
		for (int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (int i = 0; i < num_relations; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
    return PyLong_FromLongLong(num_entities);
}

static PyObject *
build_triplet_list(long long h, long long r, long long t, double y)
{
	PyObject* list = PyList_New(4);
	if (list == NULL) {
		return NULL;
	}
	PyList_SetItem(list, 0, PyLong_FromLongLong(h));
	PyList_SetItem(list, 1, PyLong_FromLongLong(r));
	PyList_SetItem(list, 2, PyLong_FromLongLong(t));
	PyList_SetItem(list, 3, PyLong_FromDouble(y));
	return list;
}

static PyObject *
gentrain_yield_triplets(PyObject *self, PyObject *args)
{
	int id = 0;
	int negRate;
	int negRelRate;
	int bernFlag;
    if (!PyArg_ParseTuple(args, "iip", &negRate, &negRelRate, &bernFlag)) {
		return NULL;
	}
	int i = rand_max(0, num_triplets);
	PyObject* list = PyList_New(0);
	if (list == NULL) {
		return NULL;
	}

	if (PyList_Append(list, build_triplet_list(trainList[i].h, trainList[i].r, trainList[i].t, 1)) == -1) {
		return NULL;
	}
	double prob = 500;
	for (int times = 0; times < negRate; times++) {
		if (bernFlag) {
			prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
		}
		if (randd(id) % 1000 < prob) {
			if (PyList_Append(list, build_triplet_list(trainList[i].h, trainList[i].r, corrupt_head(id, trainList[i].h, trainList[i].r), -1)) == -1) {
				return NULL;
			}
		} else {
			if (PyList_Append(list, build_triplet_list(corrupt_tail(id, trainList[i].t, trainList[i].r), trainList[i].r, trainList[i].t, -1)) == -1) {
				return NULL;
			}
		}
	}
	for (int times = 0; times < negRelRate; times++) {
		if (PyList_Append(list, build_triplet_list(trainList[i].h, corrupt_rel(id, trainList[i].h, trainList[i].t), trainList[i].t, -1)) == -1) {
			return NULL;
		}
	}
	return list;
}

static PyObject *
gentrain_release(PyObject *self, PyObject *args)
{
	free(trainList);
	free(trainHead);
	free(trainTail);
	free(trainRel);
	free(freqRel);
	free(freqEnt);
	free(lefHead);
	free(rigHead);
	free(lefTail);
	free(rigTail);
	free(lefRel);
	free(rigRel);
	free(left_mean);
	free(right_mean);
    return PyLong_FromLongLong(0); // random stuff here
}

static PyMethodDef GentrainMethods[] = {
	{"init_buff", gentrain_init_buff, METH_VARARGS,
	 "Allocate a buffer for that."},
    {"feed",  gentrain_feed, METH_VARARGS,
     "Takes a few parameters from Python."},
	{"freq",  gentrain_freq, METH_VARARGS,
	 "Calculate the frequence."},
	{"yield_triplets", gentrain_yield_triplets, METH_VARARGS,
	 "Yield triplet based on paramters."},
	{"release", gentrain_release, METH_VARARGS,
	 "Free resouces."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef gentrainmodule = {
   PyModuleDef_HEAD_INIT,
   "gentrain",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   GentrainMethods
};

static PyObject *GentrainError;

PyMODINIT_FUNC
PyInit_gentrain(void)
{
    PyObject *m;

    m = PyModule_Create(&gentrainmodule);
    if (m == NULL)
        return NULL;

    GentrainError = PyErr_NewException("gentrain.error", NULL, NULL);
    Py_INCREF(GentrainError);
    PyModule_AddObject(m, "error", GentrainError);
    return m;
}
