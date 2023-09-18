"""
this script contains augmentation logic for raw point clouds 
"""
import render_util
import numpy as np
import random

def null(pc): # do nothing
    return pc

def noise(pc): # uniform noise
    noise_amount = 500
    bbox_begin = np.max(pc,axis=0)
    bbox_end = np.min(pc,axis=0)
    points = np.array([[np.random.uniform(bbox_begin[i],bbox_end[i]) for i in range(3)] for j in range(noise_amount)])
    return np.concatenate([pc,points],axis=0)

def translate_x(pc):
    dx = np.random.uniform(-0.2,0.2)
    return pc + np.array([0,dx,0])

def translate_y(pc):
    dy = np.random.uniform(-0.2,0.2)
    return pc + np.array([0,0,dy])

def scale(pc):
    scale = np.random.uniform(0.7,1.2)
    pivot = render_util.center_of_mass(pc)
    return render_util.scale_pc(pc,scale,pivot=pivot)


class Augmentor:

    def __init__(self,passes):
        self.passes = passes
        self.augmentors = {}
        self._actions = []
        self._probs = []

    def add_augmentor(self,name,weight,function):
        self.augmentors[name] = (weight,function)
    
    def build(self):
        if self.is_null():
            return
        weights = np.array([e[0] for _,e in self.augmentors.items()])
        self._probs = weights / np.sum(weights)
        self._actions = [e[1] for _,e in self.augmentors.items()]

    def is_null(self): # returns true if the augmentor does nothing
        return self.passes <= 0 or \
         len(self.augmentors) == 0 or \
            (len(self.augmentors) == 1 and 'null' in self.augmentors)

    def __str__(self):
        if self.is_null():
            return "NULL augmentor"
        return '\n'.join([f"{key} : {prob}" for key,prob in zip(self.augmentors,self._probs)])

    def __call__(self,pc):
        # pc assumed to be a numpy array
        for _ in range(self.passes):
            func = random.choices(self._actions,weights = self._probs,k=1)[0]
            pc = func(pc)
        return pc
        