/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include <DecisionTree/Bagging.hpp>

int main() {
  // Dataset
  Dataset d;
  d.train.filename = "../data/covtype.arff";
  d.test.filename = "../data/covtype_test.arff";

  // Construct the ensemble model
  Bagging bc(d, 5);

  // Test
  bc.test();

  return 0;
}
