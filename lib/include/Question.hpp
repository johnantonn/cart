/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#ifndef DECISIONTREE_QUESTION_HPP
#define DECISIONTREE_QUESTION_HPP

#include <string>
#include <vector>
#include "Utils.hpp"

/**
 * Representation of a "test" on an attritbute.
 *
 * NOTE: This class can be modified.
 */
class Question {
  public:
    Question();
    Question(const int column, const int value, const MetaData& meta);

    inline const bool isNumeric() const {return isNumeric_;};
    const bool solve(int val) const; // changed to int; for training
    const bool solve(VecI example) const; // changed to vector of ints; for test
    const std::string toString(const MetaData& meta) const;

    int column_;
    int value_; // changed to int
    
  private:
    bool isNumeric_; // true if the feature to split on is numeric

};

#endif //DECISIONTREE_QUESTION_HPP
