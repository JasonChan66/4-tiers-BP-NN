// myNN.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include "./neural_network.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;


int minmaxmap(vector<double> vec) {
	double mx = *std::max_element(vec.begin(), vec.end());
	double mn = *std::min_element(vec.begin(), vec.end());
	for (auto& elem : vec) 
	{
		elem = (elem - mn) / (mx - mn);
	}
	return 1;
}

vector<double> split(string &str, char delim = ',') {
	vector<double> elems;
	std::stringstream ss;
	ss.str(str);
	string item;
	while (std::getline(ss, item, delim)) 
	{
		elems.push_back(atof(item.c_str()));
	}
	return elems;
}

bool isemptyline(string& s) {
	for (auto& c : s) 
	{
		if (!std::isspace(c)) 
		{
			return false;
		}
	}
	return true;
}

int _tmain(int argc, _TCHAR* argv[])
{
	std::ifstream data_file(argv[1]);
	std::ifstream label_file(argv[2]);
	string line;
	vector<vector<double>> total;
	vector<double> labels_total;
	while (data_file >> line) 
	{
		if (!isemptyline(line))
		 {
			vector<double> t = split(line);
			total.push_back(split(line));
		}
	}
	while (label_file >> line)
	 {
		if (!isemptyline(line)) 
		{
			labels_total.push_back(atof(line.c_str()));
		}
		minmaxmap(labels_total);
	}

	//start 10-fold cross validation
	int piecesize = labels_total.size() / 10;
	vector<double> accs;
	for (int i = 0; i < 10; i++)
	 {
		Matrix<double> validate_dataSet;
		vector<double> validate_label;
		Matrix<double> dataSet;
		vector<double> label;

		//fill validation set
		for (int j = i*piecesize; j < (i + 1)*piecesize; j++)
		 {
			validate_dataSet.inject_row(total[j]);
			validate_label.push_back(labels_total[j]);
		}
		for (int j = 0; j < i*piecesize; j++) 
		{
			dataSet.inject_row(total[j]);
			label.push_back(labels_total[j]);
		}
		for (int j = (i + 1)*piecesize; j < labels_total.size(); j++)
		 {
			dataSet.inject_row(total[j]);
			label.push_back(labels_total[j]);
		}

		FourLayerFNN myFNN(dataSet.columns(), 15, 15, 0.0001, 0.001, 1000);

		myFNN.trainFNN(dataSet, label);
		vector<double> preds = myFNN.predict(validate_dataSet);

		int hit = 0;
		for (int i = 0; i < preds.size(); i++) 
		{
			cout << "Pred: " << preds[i] << "\t" << "small_labels: " << validate_label[i] << "\t";
			if (fabs(preds[i] - validate_label[i]) < 0.5)
			 {
				cout << "HIT";
				hit++;
			}
			cout << endl;
		}
		double acc = (double)hit / dataSet.rows();
		accs.push_back(acc);
		cout << "Fold" << i << ": hit: " << hit << "\t acc: " << acc << endl;
	}

	double sum = 0;
	cout << "\naccs: " << endl;
	for (auto a : accs) 
	{
		cout << a << endl;
		sum += a;
	}
	cout << "\naverage accs: " << sum / 10 << endl;
	return 0;
}

