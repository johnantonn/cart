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

tuple<Data, Data> Calculations::partition(const Data& data, const Question& q) {
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
  ClassCounter clsCounter = classCounts(rows);
  double gini_node = gini(clsCounter, rows.size());
  // Best split for each feature
  for(int f=0; f<meta.labels.size()-1; f++){
    tuple<std::string, double> best_threshold;
    if(meta.types[f] == "CATEGORICAL"){
      best_threshold = determine_best_threshold_cat(rows, f);
    }
    else if (meta.types[f] == "NUMERIC"){
      best_threshold = determine_best_threshold_numeric(rows, f);
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

tuple<std::string, double> Calculations::determine_best_threshold_numeric(const Data& data, int col) {
  Data fData;
  double best_loss = std::numeric_limits<float>::infinity();
  int N = data.size(), nTrue=N, nFalse=0;
  std::string best_thresh, decision;
  // Construct the subset of feature and class columns
  for(int i=0; i<N; i++){
    int l = data[i].size();
    fData.push_back({data[i][col], data[i][l-1]});
  };
  // Sort based on ordinal feature
  std::sort(fData.begin(), fData.end(), [](VecS& a, VecS& b) {
    return std::stod(a[0]) < std::stod(b[0]);
  });
  // Initialize class counters
  ClassCounter clsCntTrue, clsCntFalse;
  clsCntTrue = classCounts(fData);
  // Update class counters and compute gini
  for(int i=0; i<N-1; i++){
    nTrue--;
    nFalse++;
    decision = fData[i][1];
    clsCntTrue.at(decision)--;
    if (clsCntFalse.find(decision) != std::end(clsCntFalse)) {
      clsCntFalse.at(decision)++;
    } else {
      clsCntFalse[decision] += 1;
    }
    if(fData[i][0].compare(fData[i+1][0])!=0){
      double gini_true = gini(clsCntTrue, nTrue);
      double gini_false = gini(clsCntFalse, nFalse);
      double gini_part = gini_true*((double) nTrue/N) + gini_false*((double) nFalse/N);
      if(gini_part < best_loss){
        best_loss = gini_part;
        best_thresh = fData[i+1][0];
      }
    }
  }
  return forward_as_tuple(best_thresh, best_loss);
}

tuple<std::string, double> Calculations::determine_best_threshold_cat(const Data& data, int col) {
  // Variable definitions
  Data fData;
  std::string best_thresh;
  double best_loss = std::numeric_limits<float>::infinity();
  int N = data.size();
  // Construct the subset of feature and class columns
  for(int i=0; i<N; i++){
    int l = data[i].size();
    fData.push_back({data[i][col], data[i][l-1]});
  };
  // Sort based on categorical feature
  std::sort(fData.begin(), fData.end(), [](VecS& a, VecS& b) {
    return a[0].compare(b[0]) < 0;
  });
  // Initialize class counters
  ClassCounter clsCntTrueGlobal, clsCntFalseGlobal, clsCntTrue, clsCntFalse;
  clsCntFalseGlobal = classCounts(fData);
  clsCntFalse = clsCntFalseGlobal;
  // Update class counters and compute gini
  int tmpValueCounter = 0, nTrue=0, nFalse=N;
  std::string decision;
  for(int i=0; i<N-1; i++){
    tmpValueCounter++;
    decision = fData[i][1];
    clsCntFalse.at(decision)--;
    if (clsCntTrue.find(decision) != std::end(clsCntTrue)) {
      clsCntTrue.at(decision)++;
    } else {
      clsCntTrue[decision] += 1;
    }
    if(fData[i][0].compare(fData[i+1][0])!=0){
      nTrue=tmpValueCounter;
      nFalse=N-tmpValueCounter;
      double gini_true = gini(clsCntTrue, nTrue);
      double gini_false = gini(clsCntFalse, N-nFalse);
      double gini_part = gini_true*((double) nTrue/N) + gini_false*((double) nFalse/N);
      if(gini_part < best_loss){
        best_loss = gini_part;
        best_thresh = fData[i+1][0];
      }
      tmpValueCounter=0;
      clsCntTrue=clsCntTrueGlobal;
      clsCntFalse=clsCntFalseGlobal;
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
