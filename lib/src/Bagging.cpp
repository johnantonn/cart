/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include "Bagging.hpp"

using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;

Bagging::Bagging(const DataReader& dr, const int ensembleSize, uint seed) : 
  dr_(dr), 
  ensembleSize_(ensembleSize),
  learners_({}) {
  random_number_generator.seed(seed);
  buildBag();
}


void Bagging::buildBag() {
  cpu_timer timer;
  std::vector<double> timings;
  int N = dr_.trainData()[0].size();
  int m = dr_.trainData().size();
  // Uniform distribution of integers
  std::uniform_int_distribution<int> unii(0, N-1);
  // Loop over ensemble size
  for (int i = 0; i < ensembleSize_; i++) {
    timer.start();
    // New data matrix
    Data data = std::vector<std::vector<int>>(m, std::vector<int>({}));
    int count = N;
    while(count-- > 0){
      // Index to sample
      int idx = unii(random_number_generator);
      // Due to transposition of trainData_
      for(size_t i=0; i<m; i++){
        data[i].emplace_back(dr_.trainData()[i][idx]);
      }
    }
    // Backup up original dataset
    dr_.setBaggingData(data);
    // Build decision tree on the new dataset
    DecisionTree dt(dr_);
    //dt.print();
    learners_.push_back(dt);
    // Reset the original dataset and delete the newly created one
    dr_.resetBaggingData();
    auto nanoseconds = boost::chrono::nanoseconds(timer.elapsed().wall);
    auto seconds = boost::chrono::duration_cast<boost::chrono::seconds>(nanoseconds);
    timings.push_back(seconds.count());
  }
  float avg_timing = Utils::iterators::average(std::begin(timings), std::begin(timings) + std::min(5, ensembleSize_));
  std::cout << "Average timing: " << avg_timing << std::endl;
}

void Bagging::test() const {
  TreeTest t;
  float accuracy = 0;
  for (const auto& row: dr_.testData()) {
    static size_t last = row.size() - 1;
    std::vector<int> decisions;
    for (int i = 0; i < ensembleSize_; i++) {
      const std::shared_ptr<Node> root = std::make_shared<Node>(learners_.at(i).root_);
      const auto& classification = t.classify(row, root);
      decisions.push_back(Utils::tree::getMax(classification));
    }
    int prediction = Utils::iterators::mostCommon(decisions.begin(), decisions.end());
    if (prediction == row[last])
      accuracy += 1;
  }
  std::cout << "Total accuracy: " << (accuracy / dr_.testData().size()) << std::endl;
}


