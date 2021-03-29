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
  Data true_cols = Data(data.size(),std::vector<int>({}));
  Data false_cols = Data(data.size(),std::vector<int>({}));
  for (int i=0; i<data[q.column_].size(); i++) {
    if (q.solve(data[q.column_][i])){
      for(size_t j=0; j<data.size(); j++){
        true_cols[j].push_back(data[j][i]);
      }
    }
    else{
      for(size_t j=0; j<data.size(); j++){
        false_cols[j].push_back(data[j][i]);
      }
    }
  }

  return forward_as_tuple(true_cols, false_cols);
}

tuple<const double, const Question> Calculations::find_best_split(const Data& cols, const MetaData& meta) {
  double best_gain = 0.0;  // keep track of the best information gain
  Question best_question;  // keep track of the feature / value that produced it
  int m = cols.size();
  int N = cols[0].size();
  // Unsplit node gini
  ClassCounter clsCounter = classCounts(cols[m-1]);
  double gini_node = gini(clsCounter, N);
  // Best split for each feature
  for(size_t f=0; f<m-1; f++){
    tuple<int, double> best_threshold;
    if (meta.types[f] == "NUMERIC"){
      best_threshold = determine_best_threshold_numeric(cols, f);
    }
    else if(meta.types[f] == "CATEGORICAL"){
      best_threshold = determine_best_threshold_cat(cols, f);
    }
    else {
      throw std::runtime_error("Attribute type is neither NUMERICAL nor CATEGORICAL.");
    }
    // Calculate best_threshold gain
    double gain = gini_node - std::get<1>(best_threshold);
    if(gain > best_gain){
      best_gain = gain;
      best_question = Question(f, std::get<0>(best_threshold), meta);
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

tuple<int, double> Calculations::determine_best_threshold_numeric(const Data& data, int col) {
  double best_loss = std::numeric_limits<float>::infinity();
  int m = data.size();
  int N = data[0].size();
  int best_thresh;

  // Sort mapping
  std::vector<std::size_t> index(N);
  std::iota(index.begin(), index.end(), 0);
  std::sort(index.begin(), index.end(), [&](size_t a, size_t b) { return data[col][a] < data[col][b]; });

  // Initialize class counters
  ClassCounter clsCntTrue, clsCntFalse;
  clsCntTrue = classCounts(data[m-1]);
  
  // Update class counters and compute gini
  int nTrue = N;
  for(int i=0; i<N; i++){
    nTrue--;
    int decision = data[m-1][index[i]];
    clsCntTrue.at(decision)--;
    if (clsCntFalse.find(decision) != std::end(clsCntFalse)) {
      clsCntFalse.at(decision)++;
    } else {
      clsCntFalse[decision] += 1;
    }
    if(i < N-1 && data[col][index[i]] < data[col][index[i+1]]){
      int nFalse = N - nTrue;
      double gini_true = gini(clsCntTrue, nTrue);
      double gini_false = gini(clsCntFalse, nFalse);
      double gini_part = gini_true*((double) nTrue/N) + gini_false*((double) nFalse/N);
      if(gini_part < best_loss){
        best_loss = gini_part;
        best_thresh = data[col][index[i+1]];
      }
    }
  }
  return forward_as_tuple(best_thresh, best_loss);
}

tuple<int, double> Calculations::determine_best_threshold_cat(const Data& data, int col) {
  double best_loss = std::numeric_limits<float>::infinity();
  int best_thresh;
  int N = data[0].size();
  int m = data.size();

  // Initialize class counters
  ClassCounter counterTrue;
  ClassCounter counterFalse = classCounts(data[m-1]);;
  std::unordered_map<int, ClassCounter> mapOfCountersTrue;
  std::unordered_map<int, ClassCounter> mapOfCountersFalse;
  for(size_t i=0; i<N; i++){
    int decision = data[m-1][i];
    // Check (create) class counter for true set
    if(mapOfCountersTrue.find(data[col][i]) == std::end(mapOfCountersTrue)){
      mapOfCountersTrue[data[col][i]] = counterTrue;
    }
    // Check (create) class counter for false set
    if(mapOfCountersFalse.find(data[col][i]) == std::end(mapOfCountersFalse)){
      mapOfCountersFalse[data[col][i]] = counterFalse;
    }
    // Update
    mapOfCountersFalse.at(data[col][i]).at(decision)--;
    if (mapOfCountersTrue.at(data[col][i]).find(decision) != std::end(mapOfCountersTrue.at(data[col][i]))) {
      mapOfCountersTrue.at(data[col][i]).at(decision)++;
    } else {
      mapOfCountersTrue.at(data[col][i])[decision] += 1;
    }
  }

  // Compute gini for each value
  for(const auto& n: mapOfCountersTrue) {
    int nTrue = 0;
    for(const auto& m: mapOfCountersTrue.at(n.first)) {
      nTrue += m.second;
    }
    int nFalse = N - nTrue;
    double gini_true = gini(n.second, nTrue);
    double gini_false = gini(mapOfCountersFalse.at(n.first), nFalse);
    double gini_part = gini_true*((double) nTrue/N) + gini_false*((double) nFalse/N);
    if(gini_part < best_loss){
      best_loss = gini_part;
      best_thresh = n.first;
    }
  }
  return forward_as_tuple(best_thresh, best_loss);
}

const ClassCounter Calculations::classCounts(const VecI& classVec) {
  ClassCounter counter;
  for (int i=0; i<classVec.size(); i++) {
    const int decision = classVec[i];
    if (counter.find(decision) != std::end(counter)) {
      counter.at(decision)++;
    } else {
      counter[decision] += 1;
    }
  }
  return counter;
}
