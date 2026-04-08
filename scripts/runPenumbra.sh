#!/bin/bash

#python3 lithobench/train.py -m lithobench/ilt/fpga_neuralilt.py -a FPGANeuralILT -i 512 -t ILT -s MetalSet -n 8 -b 12 -p True
PENUMBRA_TEACHER=saved/MetalSet_NeuralILT/net.pth python3 lithobench/train.py -m lithobench/ilt/fpga_neuralilt.py -a FPGANeuralILT -i 512 -t ILT -s MetalSet -n 8 -b 12 -p True
# python3 lithobench/train.py -m lithobench/ilt/fpga_neuralilt.py -a FPGANeuralILT -i 512 -t ILT -s ViaSet -n 2 -b 12 -p True

# python3 lithobench/test.py -m lithobench/ilt/fpga_neuralilt.py -a FPGANeuralILT -i 512 -t ILT -s MetalSet -l saved/MetalSet_FPGANeuralILT/net.pth --shots
# python3 lithobench/test.py -m lithobench/ilt/fpga_neuralilt.py -a FPGANeuralILT -i 512 -t ILT -s ViaSet -l saved/ViaSet_FPGANeuralILT/net.pth --shots
# python3 lithobench/test.py -m lithobench/ilt/fpga_neuralilt.py -a FPGANeuralILT -i 512 -t ILT -s StdMetal -l saved/MetalSet_FPGANeuralILT/net.pth --shots
# python3 lithobench/test.py -m lithobench/ilt/fpga_neuralilt.py -a FPGANeuralILT -i 512 -t ILT -s StdContact -n 16 -b 12 -l saved/ViaSet_FPGANeuralILT/net.pth --shots
