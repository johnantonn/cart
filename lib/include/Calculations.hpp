/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#ifndef DECISIONTREE_CALCULATIONS_HPP
#define DECISIONTREE_CALCULATIONS_HPP

#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>
#include <boost/timer/timer.hpp>
#include "Question.hpp"
#include "Utils.hpp"

using ClassCounter = std::unordered_map<int, int>;

namespace Calculations {

std::tuple<Data, Data> partition(const Data &data, const Question &q);

const double gini(const ClassCounter& counts, double N);

std::tuple<const double, const Question> find_best_split(const Data &cols, const MetaData &meta);

std::tuple<int, double> determine_best_threshold_numeric(const Data &data, int col);

std::tuple<int, double> determine_best_threshold_cat(const Data &data, int col);

const ClassCounter classCounts(const VecI& classVec);

void zip(const std::vector<int> &a, const std::vector<int> &b, std::vector<std::pair<int,int>> &zipped);

void unzip(const std::vector<std::pair<int, int>> &zipped, std::vector<int> &a, std::vector<int> &b);

} // namespace Calculations

#endif //DECISIONTREE_CALCULATIONS_HPP
