/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include <DecisionTree/DecisionTree.hpp>

int main() {
  // Dataset
  Dataset d;
  d.train.filename = "../data/covtype.arff";
  d.test.filename = "../data/covtype_test.arff";

  // Construct the decision tree
  DecisionTree dt(d);

  // Print the decision tree
  //dt.print();

  // Test
  dt.test();

  return 0;
}
