import os
import subprocess

datafile = 'data/benchmark_GAN_generated_cosmo_maps_64.npy'
output_size = 64
epoch = 3
flip_labels = 0.01
z_dim = 64
nd_layers = 4
ng_layers = 4
gf_dim = 64
df_dim = 64
transpose_matmul_b = False
verbose = 'False'
arch = 'default' #default, KNL or HSW

experiment = 'benchmark_scan'

command = 'python dcgan/scan.py --dataset cosmo --datafile %s '\
          '--output_size %i --flip_labels %f --experiment %s '\
          '--epoch %i --z_dim %i --arch %s '\
          '--nd_layers %i --ng_layers %i --gf_dim %i --df_dim %i '\
          '--transpose_matmul_b %s --verbose %s'%(datafile, output_size, flip_labels, experiment,\
                                                 epoch, z_dim, arch,\
                                                 nd_layers, ng_layers, gf_dim, df_dim,\
                                                 transpose_matmul_b, verbose)
subprocess.call(command.split())
