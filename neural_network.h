#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <vector>
#include <exception>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iostream>
using std::cout;
using std::endl;
using std::vector;
using std::string;

class MisException : std::exception {
public:
	MisException(string str)
		: std::exception(), msg(str) {}

private:
	string msg;
};


template <typename T>
class myMatrix {
private:
	vector<vector<T>> data;
	int row_count;
	int column_count;

public:
	myMatrix<T>(int r = 0, int c = 0);
	myMatrix<T> &inject_row(vector<T> row);
	myMatrix<T> &inject_column(vector<T> row);
	vector<T> get_row(int nrow);
	vector<T> get_column(int ncolumn);
	int rows();
	int columns();
	bool random_init(int nrow, int ncolumn);
	bool random_init(int nrow, int ncolumn, T range_low, T range_high);
	void update(int r, int c, T content);

};

template <typename T>
myMatrix<T>::myMatrix(int r, int c)
: row_count(r), column_count(c) {
	for (int i = 0; i < r; i++) {
		vector<T> tmp;
		for (int j = 0; j < c; j++) {
			tmp.push_back(T());
		}
		data.push_back(tmp);
	}
}

template <typename T>
myMatrix<T> &myMatrix<T>::inject_column(vector<T> column) {
	if (this->row_count != 0 && column.size() != this->row_count) {
		throw MisException("Cannot inject column![ERR_ROW_MISMATCH]");
	}
	for (int i = 0; i < column.size(); i++) {
		data[i].push_back(column[i]);
	}
	this->column_count++;
	if (this->row_count == 0)
		this->row_count = column.size();
	return *this;
}
template <typename T>
myMatrix<T> &myMatrix<T>::inject_row(vector<T> row) {
	if (this->column_count != 0 && row.size() != this->column_count) {
		throw MisException("Cannot inject row![ERR_COLUMN_MISMATCH]");
	}
	data.push_back(row);
	this->row_count++;
	if (this->column_count == 0)
		this->column_count = row.size();
	return *this;
}



template <typename T>
vector<T> myMatrix<T>::get_column(int ncolumn) {
	if (!(ncolumn < this->column_count)) {
		throw MisException("Cannot get column[ERR_COL_INDEX_EXCEED]");
	}
	vector<T> col;
	for (int i = 0; i < this->row_count; i++) {
		col.push_back(data[i][ncolumn]);
	}
	return col;
}

template <typename T>
vector<T> myMatrix<T>::get_row(int nrow) {
	if (!(nrow < this->row_count)) {
		throw MisException("Cannot get row[ERR_ROW_INDEX_EXCEED]");
	}
	return data[nrow];
}


template <typename T>

void myMatrix<T>::update(int r, int c, T content) {
	if (r >= this->row_count || c >= this->column_count) {
		throw MisException("Fail update [ERR_INDEX_EXCEED]");
	}
	this->data[r][c] = content;
}

template <typename T>

template <typename T>
int myMatrix<T>::columns() {
	return this->column_count;
}

template <typename T>
bool myMatrix<T>::random_init(int nrow, int ncolumn) {
	random_init(nrow, ncolumn, 0, 1);
	return true;
}

int myMatrix<T>::rows() {
	return this->row_count;
}


template <typename T>
bool myMatrix<T>::random_init(int nrow, int ncolumn, T range_low, T range_high) {
	if (range_high <= 0)
		range_high = 1;
	if (range_low < 0)
		range_low = 0;
	this->data.erase(this->data.begin(), this->data.end());
	srand(time(NULL)); //initialize random seed
	for (int i = 0; i < nrow; i++) {
		vector<T> tmp;
		for (int j = 0; j < ncolumn; j++) {
			double random_number = (rand() % 10) / 1000.0f;
			tmp.push_back(random_number);
		}
		this->data.push_back(tmp);
	}
	this->row_count = nrow;
	this->column_count = ncolumn;
	return true;
}


class myNN {
public:
	myNN(int l1_count, int l2_count, int l3_count,
		double err_to_stop, double learning_rate, int iter_max);
	int trainFNN(myMatrix<double> &trainingSet, vector<double> &label);
	vector<double> predict(myMatrix<double> &dataSet);
	void outputParameter();

private:
	int nl1;
	int nl2;
	int nl3;

	double err;
	double alpha;
	int iter_max;

	myMatrix<double> weight_ij;
	myMatrix<double> weight_jo;
	myMatrix<double> weight_ok;

	double compJ(myMatrix<double> &trainingSet, vector<double> &label);
	vector<double> compute_l2_Y(vector<double> r);
	vector<double> compute_l3_Y(vector<double> r);
	double computeY(vector<double> r);
};



vector<double> myNN::compute_l2_Y(vector<double> r) {
	int l2num;
	vector<double> result;
	for (l2num = 0; l2num < this->nl2; l2num++) {
		double sum = 0;
		for (int j = 0; j < this->weight_ij.rows(); j++) {
			sum += this->weight_ij.get_row(j)[l2num] * r[j];
		}
		sum = 1 / (1 + exp(-sum));
		result.push_back(sum);
	}
	return result;
}

vector<double> myNN::compute_l3_Y(vector<double> r) {
	int l3num;
	vector<double> result;
	for (l3num = 0; l3num < this->nl3; l3num++) {
		double sum = 0;
		for (int j = 0; j < this->weight_jo.rows(); j++) {
			sum += this->weight_jo.get_row(j)[l3num] * r[j];
		}
		sum = 1 / (1 + exp(-sum));
		result.push_back(sum);
	}
	return result;
}

void myNN::outputParameter() {
	cout << "weight_ij: " << endl;
	for (int i = 0; i < this->weight_ij.rows(); i++) {
		auto& r = this->weight_ij.get_row(i);
		for (auto j : r) {
			cout << j << "\t";
		}
		cout << endl;
	}
	cout << endl;

	cout << "weight_jo: " << endl;
	for (int i = 0; i < this->weight_jo.rows(); i++) {
		auto& r = this->weight_jo.get_row(i);
		for (auto j : r) {
			cout << j << "\t";
		}
		cout << endl;
	}
	cout << endl;

	for (int i = 0; i < this->weight_ok.rows(); i++) {
		auto& r = this->weight_ok.get_row(i);
		for (auto j : r) {
			cout << j << "\t";
		}
		cout << endl;
	}
}

double myNN::computeY(vector<double> r) {
	double sum = 0;
	for (int j = 0; j < this->weight_ok.rows(); j++) {
		sum += this->weight_ok.get_row(j)[0] * r[j];
	}
	return 1 / (1 + exp(-sum));
}

myNN::myNN(int l1_count, int l2_count, int l3_count,
	double err_to_stop, double learning_rate, int iter_max)
	: nl1(l1_count), nl2(l2_count), nl3(l3_count), err(err_to_stop),
	alpha(learning_rate), iter_max(iter_max) {
	this->weight_ij.random_init(l1_count, l2_count);
	this->weight_jo.random_init(l2_count, l3_count);
	this->weight_ok.random_init(l3_count, 1);
}

int myNN::trainFNN(myMatrix<double> &trainingSet, vector<double> &label) {
	if (trainingSet.rows() != label.size()) {
		return -1;
	}
	int _i = 0;
	double e = this->err + 1;
	double old_J = compJ(trainingSet, label);
	double new_J = 0;
	while (_i++ < iter_max && e > this->err) {
		int i, j, k, l, m;
		double tmp;

		for (i = 0; i < trainingSet.rows(); i++) {
			vector<double> current_row = trainingSet.get_row(i);
			double yd = label[i];

			vector<double> l1_Y = current_row;
			vector<double> l2_Y = compute_l2_Y(current_row);
			vector<double> l3_Y = compute_l3_Y(l2_Y);
			double y = computeY(l3_Y);

			double ek = yd - y;
			double delta_k = ek * y * (1 - y);

			vector<double> delta_w_ok;
			for (k = 0; k < this->nl3; k++) {
				delta_w_ok.push_back(this->alpha * l3_Y[k] * delta_k);
			}

			myMatrix<double> delta_w_jo;
			for (k = 0; k < this->nl2; k++) {
				vector<double> delta_w_jo_row;
				for (l = 0; l < this->nl3; l++) {
					tmp = weight_ok.get_row(l)[0];                 
					tmp = tmp * delta_k * l3_Y[l] * (1 - l3_Y[l]);
					tmp = this->alpha * l2_Y[k] * tmp;            
					delta_w_jo_row.push_back(tmp);
				}
				delta_w_jo.inject_row(delta_w_jo_row);
			}


			myMatrix<double> delta_w_ij;
			for (k = 0; k < this->nl1; k++) {

				vector<double> delta_w_ij_row;
				for (l = 0; l < this->nl2; l++) {
					double sum = 0;
					for (m = 0; m < this->nl3; m++) {
						tmp = weight_ok.get_row(m)[0]; 
						tmp = tmp * delta_k * l3_Y[m] * (1 - l3_Y[m]);
						tmp = tmp * weight_jo.get_row(l)[m];
						sum += tmp;
					}
					sum *= l2_Y[l] * (1 - l2_Y[l]); 
					sum *= this->alpha * l1_Y[k];
					delta_w_ij_row.push_back(sum);
				}
				delta_w_ij.inject_row(delta_w_ij_row);
			}

			for (j = 0; j < delta_w_ij.rows(); j++) {
				const vector<double> &r = delta_w_ij.get_row(j);
				for (k = 0; k < delta_w_ij.columns(); k++) {
					this->weight_ij.update(j, k,
						this->weight_ij.get_row(j)[k] + r[k]);
				}
			}

			for (j = 0; j < delta_w_jo.rows(); j++) {
				const vector<double> &r = delta_w_jo.get_row(j);
				for (k = 0; k < delta_w_jo.columns(); k++) {
					this->weight_jo.update(j, k,
						this->weight_jo.get_row(j)[k] + r[k]);
				}
			}

			for (j = 0; j < delta_w_ok.size(); j++) {
				this->weight_ok.update(j, 0,
					this->weight_ok.get_row(j)[0] + delta_w_ok[j]);
			}
		}

		new_J = compJ(trainingSet, label);
		e = fabs(new_J - old_J);

		std::cout << "e: " << e << ", J: " << new_J << std::endl;
		old_J = new_J;
	}

	return 0;
}

double myNN::compJ(myMatrix<double> &trainingSet, vector<double> &label){

	vector<double> predict_result = this->predict(trainingSet);
	double sum = 0;
	for (int i = 0; i<predict_result.size(); i++){
		sum += ((predict_result[i] - label[i]) * (predict_result[i] - label[i]));
	}
	return sum;
}


vector<double> myNN::predict(myMatrix<double> &dataSet) {
	int i;
	vector<double> result;
	for (i = 0; i < dataSet.rows(); i++) {
		vector<double> current_row = dataSet.get_row(i);
		vector<double> l2_Y = compute_l2_Y(current_row);
		vector<double> l3_Y = compute_l3_Y(l2_Y);
		double y = computeY(l3_Y);
		result.push_back(y);
	}
	return result;
}



#endif
