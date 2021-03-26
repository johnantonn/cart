/*
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission. 
 * Written by Pieter Robberechts, 2019
 */

#include "DecisionTree.hpp"
#include "Calculations.hpp"

using std::make_shared;
using std::shared_ptr;
using std::string;
using boost::timer::cpu_timer;

DecisionTree::DecisionTree(const DataReader& dr) : root_(Node()), dr_(dr) {
  std::cout << "Start building tree." << std::endl; cpu_timer timer;
  root_ = buildTree(dr_.trainData(), dr_.metaData());
  std::cout << "Done. " << timer.format() << std::endl;
}

const Node DecisionTree::buildTree(const Data& rows, const MetaData& meta) {
  auto [gain, question] = Calculations::find_best_split(rows, meta);
  if(gain == 0){
    ClassCounter clsCounter = Calculations::classCounts(rows);
    return Node(Leaf(clsCounter));
  }
  else {
    auto [true_rows, false_rows] = Calculations::partition(rows, question);
    std::cout << question.toString(meta) << std::endl;
    std::cout << "True branch: " << true_rows.size() << ", False branch: " << false_rows.size() << std::endl;
    Node trueBranch = buildTree(true_rows, meta);
    true_rows.clear();
    Node falseBranch = buildTree(false_rows, meta);
    false_rows.clear();
    return Node(trueBranch, falseBranch, question);
  }
}

void DecisionTree::print() const {
  print(make_shared<Node>(root_));
}

void DecisionTree::print(const shared_ptr<Node> root, string spacing) const {
  if (bool is_leaf = root->leaf() != nullptr; is_leaf) {
    const auto &leaf = root->leaf();
    std::cout << spacing + "Predict: "; Utils::print::print_map(leaf->predictions());
    return;
  }
  std::cout << spacing << root->question().toString(dr_.metaData()) << "\n";

  std::cout << spacing << "--> True: " << "\n";
  print(root->trueBranch(), spacing + "   ");

  std::cout << spacing << "--> False: " << "\n";
  print(root->falseBranch(), spacing + "   ");
}

void DecisionTree::test() const {
  TreeTest t(dr_.testData(), dr_.metaData(), root_);
}
