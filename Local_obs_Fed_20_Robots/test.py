# #list of tuples of key-value pairs
# import math
# import numpy as np
# from math import sqrt
# robotID = 0
# imgRecSize = 10
# task = dict([("type", "imageRecognition"), ("size", imgRecSize),("source", robotID)])
# print(task["size"])
# channel_gains = [[0 for m in range(14)] for n in range(10)]
# target1 = [[0 for x in range(11)] for y in range(2)]
#
#
# a = np.array([1, 2, 3, 4])
# b = np.array([10, 20, 30, 4])
#
# c = np.array([122, 22, 33, 4])
# action_Robots = np.random.uniform(-2, 2)
# p = np.concatenate([a, b])
#
# if np.clip(action_Robots, -1.0, 1.0) < -0.7:
#     howManyPortion = 1
# print(np.clip(action_Robots, -1.0, 1.0))
# !/usr/bin/env python

import numpy as np
import random

#
#x = [[100, 100], [100, 300], [100,500], [100, 700], [100, 900],
#              [300, 100], [300, 300], [300,500], [300, 700], [300, 900],
#              [500, 100], [500, 300], [500,500], [500, 700], [500, 900],
#              [700, 100], [700, 300], [700,500], [700, 700], [700, 900],
#              [900, 100], [900, 300], [900,500], [900, 700], [900, 900]]
#
#numlist = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#print(np.average(numlist))
# for i in range(25):
#     a = np.random.uniform(0.01, 0.05, 25)
#     chosen_numb = random.choice(numlist)
#     numlist.remove(chosen_numb)
#
#     # posX = x[chosen_numb][0]
#     # posY = x[chosen_numb][1]
#
#     print(a)

'''
pso.py
A simple implementation of the Particle Swarm Optimisation Algorithm.
Uses Numpy for matrix operations.
Pradeep Gowda 2009-03-16
'''
#
# from numpy import array
# from random import random
# from math import sin, sqrt
#
# iter_max = 10000
# pop_size = 100
# dimensions = 2
# c1 = 2
# c2 = 2
# err_crit = 0.00001
#
#
# class Particle:
#     pass
#
#
# def f6(param):
#     '''Schaffer's F6 function'''
#     para = param * 10
#     para = param[0:2]
#     num = (sin(sqrt((para[0] * para[0]) + (para[1] * para[1])))) * \
#           (sin(sqrt((para[0] * para[0]) + (para[1] * para[1])))) - 0.5
#     denom = (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1]))) * \
#             (1.0 + 0.001 * ((para[0] * para[0]) + (para[1] * para[1])))
#     f6 = 0.5 - (num / denom)
#     errorf6 = 1 - f6
#     return f6, errorf6;
#
#
# # initialize the particles
# particles = []
# for i in range(pop_size):
#     p = Particle()
#     p.params = array([random() for i in range(dimensions)])
#     p.fitness = 0.0
#     p.v = 0.0
#     particles.append(p)
#
# # let the first particle be the global best
# gbest = particles[0]
# err = 999999999
# while i < iter_max:
#     for p in particles:
#         fitness, err = f6(p.params)
#         if fitness > p.fitness:
#             p.fitness = fitness
#             p.best = p.params
#
#         if fitness > gbest.fitness:
#             gbest = p
#         v = p.v + c1 * random() * (p.best - p.params) \
#             + c2 * random() * (gbest.params - p.params)
#         p.params = p.params + v
#
#     i += 1
#     if err < err_crit:
#         break
#     # progress bar. '.' = 10%
#     if i % (iter_max / 10) == 0:
#         print(        '.')
#
# print('\nParticle Swarm Optimisation\n')
# print('PARAMETERS\n', '-' * 9)
# print('Population size : ', pop_size)
# print('Dimensions      : ', dimensions)
# print('Error Criterion : ', err_crit)
# print('c1              : ', c1)
# print('c2              : ', c2)
# print('function        :  f6')
#
# print('RESULTS\n', '-' * 7)
# print('gbest fitness   : ', gbest.fitness)
# print('gbest params    : ', gbest.params)
# print('iterations      : ', i + 1)
# ## Uncomment to print particles
# # for p in particles:
# #    print 'params: %s, fitness: %s, best: %s' % (p.params, p.fitness, p.best)
import struct

def binfracdigits(n, bits, fracbits):
    t = int(n * (2 ** fracbits))
    s = bin(t & int("1" * bits, 2))[2:]
    print( ("{0:0>%s}" % (bits)).format(s))
def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
v = 0.88*8

print(v, "\n","  %b", binary(v))
binfracdigits(v, 16, 12)