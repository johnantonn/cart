/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include "Question.hpp"
#include "Utils.hpp"

using std::string;
using std::vector;

Question::Question(): column_(0), value_(0), isNumeric_(true){}
Question::Question(const int column, const int value, const MetaData& meta) : column_(column), value_(value), isNumeric_(meta.types[column_]=="NUMERIC")
 {}

const bool Question::solve(int val) const {
  if (isNumeric()) {
    return val >= value_;
  } else {
    return val == value_;
  }
}

const bool Question::solve(VecI example) const {
  const int& val = example[column_];
    if (isNumeric()) {
    return val >= value_;
  } else {
    return val == value_;
  }
}

const string Question::toString(const MetaData& meta) const {
  string condition = ">=";
  string val = std::to_string(value_);
  if (!isNumeric()){
    condition = "==";
    val = meta.dMapIS.at(meta.labels[column_]).at(value_);
  }
  return "Is " + meta.labels[column_] + " " + condition + " " + val + "?";
}
