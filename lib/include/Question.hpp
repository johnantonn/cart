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
    const bool solve(int val) const;
    const bool solve(VecI example) const;
    const std::string toString(const MetaData& meta) const;

    int column_;
    int value_;
    
  private:
    bool isNumeric_;

};

#endif //DECISIONTREE_QUESTION_HPP
