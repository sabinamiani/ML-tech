#!/bin/sh

pip3 install numpy 
pip3 install pandas

echo "HW1 Decision Tree Alg by Sabina Miani"
echo "Insert dt max depth: "

read max_depth

DT=$(python3 hw1_decision_tree.py $max_depth)

echo "Car Decision Tree with depth $max_depth: "
echo "$DT"
