/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include <cmath>
#include <algorithm>
#include <iterator>
#include "Calculations.hpp"
#include "Utils.hpp"

using std::tuple;
using std::pair;
using std::forward_as_tuple;
using std::vector;
using std::string;
using std::unordered_map;

tuple<const Data, const Data> Calculations::partition(const Data& data, const Question& q) {
  Data true_rows;
  Data false_rows;
  
  for (const auto &row: data) {
    if (q.solve(row))
      true_rows.push_back(row);
    else
      false_rows.push_back(row);
  }

  return forward_as_tuple(true_rows, false_rows);
}

tuple<const double, const Question> Calculations::find_best_split(const Data& rows, const MetaData& meta) {
  double best_gain = 0.0;  // keep track of the best information gain
  auto best_question = Question();  // keep track of the feature / value that produced it
  int N = rows.size(); // number of data points
  // Calculate node's Gini impurity
  // Calculate class counts
  ClassCounter clsCounter = classCounts(rows);
  // Calculate total gini
  double gini_node = gini(clsCounter, N);
  //std::cout << "Node Gini: " << gini_node << std::endl;
  // Best split for each feature
  for(int f=0; f<meta.labels.size()-1; f++){
    tuple<std::string, double> best_threshold;
    // std::cout << meta.labels[f] << ", " << meta.types[f] << std::endl;
    if(meta.types[f] == "CATEGORICAL"){
      best_threshold = determine_best_threshold_cat(rows, f);
    }
    else if (meta.types[f] == "NUMERIC"){
      best_threshold = determine_best_threshold_numeric1(rows, f);
    }
    else {
      throw std::runtime_error("Attribute type is neither NUMERICAL nor CATEGORICAL.");
    }
    // Calculate best_threshold gain
    double gain = gini_node - std::get<1>(best_threshold);
    if(gain > best_gain){
      best_gain = gain;
      best_question = Question(f, std::get<0>(best_threshold));
    }
  }
  return forward_as_tuple(best_gain, best_question);
}

const double Calculations::gini(const ClassCounter& counts, double N) {
  double impurity = 1.0;
  for(const auto& [key, value]: counts){
    impurity -= pow(value/N, 2);
  }
  return impurity;
}

tuple<std::string, double> Calculations::determine_best_threshold_numeric1(const Data& data, int col) {
  std::cout << "Ordinal feature: " << col << std::endl;
  double best_loss = std::numeric_limits<float>::infinity();
  std::string best_thresh;
  int N = data.size();
  // Check for N=0
  if(N==0){return forward_as_tuple(best_thresh, best_loss);};
  std::cout << "Number of data points: " << N << std::endl;
  Data fData;
  // Construct the subset of feature and class columns
  for(int i=0; i<N; i++){
    int l = data[i].size();
    fData.push_back({data[i][col], data[i][l-1]});
  };
  // Sort data based on feature of col (numeric/ordinal)
  std::sort(fData.begin(), fData.end(), [](VecS& a, VecS& b) {
    return std::stod(a[0]) < std::stod(b[0]);
  });
  if(N==2){
    for(int i=0; i<2; i++){
      for(int j=0; j<2; j++){
        std::cout << fData[i][j] << ", ";
      }
      std::cout << std::endl;
    }
  }
  // Trivial split (according to first value)
  int nTrue=fData.size(), nFalse=0;
  // Calculate class counters once
  ClassCounter clsCntTrue = classCounts(fData);
  ClassCounter clsCntFalse;
  // Gini
  double gini_true = gini(clsCntTrue, nTrue);
  if(N == 2){
    std::cout << "Calculating gini for false class..." << nFalse << std::endl;
  }
  double gini_false = gini(clsCntFalse, nFalse);
  if(N==2){
    std::cout << "Calculated gini index for false class:" << gini_false << std::endl;
  }
  std::cout << "N=" << N << ", " << "nTrue=" << nTrue << ", nFalse=" << nFalse << ", gini_true="<< gini_true << ", gini_false=" << gini_false << std::endl;
  double gini_part = gini_true*((double) nTrue/N) + gini_false*((double) nFalse/N);
  if(gini_part < best_loss){
    best_loss = gini_part;
    best_thresh = fData[0][0];
  }
  // Update counters
  for(int i=0; i<N; i++){
    nTrue--;
    nFalse++;
    std::string decision = fData[i][1];
    clsCntTrue.at(decision)--;
    if (clsCntFalse.find(decision) != std::end(clsCntFalse)) {
      clsCntFalse.at(decision)++;
    } else {
      clsCntFalse[decision] += 1;
    }
    gini_true = gini(clsCntTrue, nTrue);
    gini_false = gini(clsCntFalse, nFalse);
    gini_part = gini_true*((double) nTrue/N) + gini_false*((double) nFalse/N);
    if(gini_part < best_loss){
      best_loss = gini_part;
      best_thresh = fData[i][0];
      //std::cout << "Index: " << i << ", N = " << N << " = " << nTrue << " + " << nFalse << ", gini: " << gini_part << std::endl;
    }
  }
  std::cout << "best_loss: " << best_loss << ", best_thresh: " << best_thresh << std::endl;
  return forward_as_tuple(best_thresh, best_loss);
}

tuple<std::string, double> Calculations::determine_best_threshold_numeric(const Data& data, int col) {
  std::cout << "Numerical feature: " << col << std::endl;
  std::cout << "Number of data points: " << data.size() << std::endl;
  double best_loss = std::numeric_limits<float>::infinity();
  std::string best_thresh;
  // For each value
  for(int i=0; i<data.size(); i++){
    Question q(col, data[i][col]);
    auto [true_rows, false_rows] = partition(data, q);
    ClassCounter clsCntTrue = classCounts(true_rows);
    ClassCounter clsCntFalse = classCounts(false_rows);
    double gini_true = gini(clsCntTrue, true_rows.size());
    double gini_false = gini(clsCntFalse, false_rows.size());
    double gini_part = gini_true*((double) true_rows.size()/data.size()) + gini_false*((double) false_rows.size()/data.size());
    if(gini_part < best_loss){
      best_loss = gini_part;
      best_thresh = data[i][col];
      //std::cout << "Index: " << i << ", N = " << data.size() << " = " << true_rows.size() << " + " << false_rows.size() << ", gini: " << gini_part << std::endl;
    }
  }
  return forward_as_tuple(best_thresh, best_loss);
}

tuple<std::string, double> Calculations::determine_best_threshold_cat(const Data& data, int col) {
  std::cout << "Categorical feature: " << col << std::endl;
  std::cout << "Number of data points: " << data.size() << std::endl;
  double best_loss = std::numeric_limits<float>::infinity();
  std::string best_thresh;
  // For each value
  for(int i=0; i<data.size(); i++){
    Question q(col, data[i][col]);
    auto [true_rows, false_rows] = partition(data, q);
    ClassCounter clsCntTrue = classCounts(true_rows);
    ClassCounter clsCntFalse = classCounts(false_rows);
    double gini_true = gini(clsCntTrue, true_rows.size());
    double gini_false = gini(clsCntFalse, false_rows.size());
    double gini_part = gini_true*((double) true_rows.size()/data.size()) + gini_false*((double) false_rows.size()/data.size());
    if(gini_part < best_loss){
      best_loss = gini_part;
      best_thresh = data[i][col];
    }
  }
  return forward_as_tuple(best_thresh, best_loss);
}


const ClassCounter Calculations::classCounts(const Data& data) {
  ClassCounter counter;
  for (const auto& rows: data) {
    const string decision = *std::rbegin(rows);
    if (counter.find(decision) != std::end(counter)) {
      counter.at(decision)++;
    } else {
      counter[decision] += 1;
    }
  }
  return counter;
}
