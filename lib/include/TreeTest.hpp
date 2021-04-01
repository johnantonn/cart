/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#ifndef DECISIONTREE_TREETEST_HPP
#define DECISIONTREE_TREETEST_HPP

#include "Node.hpp"
#include "Utils.hpp"

using ClassCounterScaled = std::unordered_map<int, std::string>; // changed to int (hash)

class TreeTest {
  public:
    TreeTest() = default;
    // Added a MetaData argument that holds the mappings of strings to integers
    TreeTest(const Data& testData, const MetaData& meta, const Node &root);
    ~TreeTest() = default;

    // Changed VecS to VecI due to mapping of strings to integers
    const ClassCounter classify(const VecI& row, std::shared_ptr<Node> node) const;

  private:
    // Added a MetaData argument that holds the mappings of strings to integers
    void printLeaf(ClassCounter counts, const MetaData& meta) const;
    // Added a MetaData argument that holds the mappings of strings to integers
    void test(const Data& testing_data, const MetaData& meta, std::shared_ptr<Node> tree) const;
};

#endif //DECISIONTREE_TREETEST_HPP
