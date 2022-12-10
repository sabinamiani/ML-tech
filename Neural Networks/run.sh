#!/bin/sh

pip3 install numpy
pip3 install torch 

echo "HW5 by Sabina Miani"
echo ""

echo "Three Layer NN - backpropogation alg"
Q4=$(python3 nn3layer.py)
echo "$Q4"

echo ""
echo "BONUS PyTorch Question"
Q4=$(python3 pytorchimp.py)
echo "$Q4"
