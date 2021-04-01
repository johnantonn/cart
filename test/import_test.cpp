/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include <DecisionTree/DataReader.hpp>

int main(void) {
  // Dataset
  Dataset d;
  d.train.filename = "../data/covtype.arff";
  d.test.filename = "../data/covtype_test.arff";

  // Import dataset
  DataReader dr(d);

  return 0;
}
