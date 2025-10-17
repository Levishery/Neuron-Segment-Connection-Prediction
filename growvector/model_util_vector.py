# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
import math
import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


class VectorDatasetConfig(object):
    def __init__(self, test_target=None):
        self.num_yaw_bin = 8
        self.num_pitch_bin = 6
        self.biological_radius = [0.3216]
        self.biological_dist = [500]
        if test_target == 'dist':
            self.biological_dist = [680, 640, 600, 560, 520, 500, 480, 440, 400, 360, 320, 280, 240, 220, 200]
        if test_target == 'radius':
            self.biological_radius = [0.40, 0.36, 0.32, 0.28, 0.26, 0.24, 0.20, 0.16, 0.12]
        self.terminal_thresh = 0.8
        self.biological_move_back = [0]

    def angle2class(self, angle, type):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from  
            class center angle to current angle.
           
            yaw angle is from 0-2pi (or -pi~pi), class center at pi/N, 1*(2pi/N)+pi/N, 2*(2pi/N)+pi/N ...  (N-1)*(2pi/N)+pi/N
            pitch angle is from 0-pi (or -pi/2~pi/2), class center at pi/2M, 1*(pi/M)+pi/2M, 2*(pi/M)+pi/2M ...  (N-1)*(pi/M)+pi/2M
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle
        '''
        if type == 'yaw':
            num_class = self.num_yaw_bin
            angle = min(angle % (2 * np.pi), 2 * np.pi - 1e-6)
            assert (angle >= 0 and angle < 2 * np.pi)
            angle_per_class = 2 * np.pi / float(num_class)
            class_id = int(angle / angle_per_class)
            residual_angle = angle - (class_id * angle_per_class + angle_per_class / 2)
            if class_id >= self.num_yaw_bin:
                pdb.set_trace()
            return class_id, residual_angle
        elif type == 'pitch':
            # pitch angle ranges from 0~pi, which means they don't have overlap of 0/2pi
            num_class = self.num_pitch_bin
            angle = min(angle + np.pi/2, np.pi - 1e-6)
            assert (angle >= 0 and angle < np.pi)
            angle_per_class = np.pi / float(num_class)
            class_id = int(angle / angle_per_class)
            residual_angle = angle - (class_id * angle_per_class + angle_per_class / 2)
            if class_id >= self.num_pitch_bin:
                pdb.set_trace()
            return class_id, residual_angle

    def class2angle(self, pred_cls, residual, type, to_label_format=True):
        if type == 'yaw':
            ''' Inverse function to angle2class '''
            num_class = self.num_yaw_bin
            angle_per_class = 2 * np.pi / float(num_class)
            angle_center = pred_cls * angle_per_class + angle_per_class / 2
            angle = angle_center + residual
            if to_label_format and angle > np.pi:
                angle = angle - 2 * np.pi
            return angle
        elif type == 'pitch':
            ''' Inverse function to angle2class '''
            num_class = self.num_pitch_bin
            angle_per_class = np.pi / float(num_class)
            angle_center = pred_cls * angle_per_class + angle_per_class / 2
            angle = angle_center + residual
            if to_label_format:
                angle = angle - np.pi/2
            return angle

    def param2obb(self, center, yaw_class, yaw_residual, pitch_class, pitch_residual):
        yaw_angle = self.class2angle(yaw_class, yaw_residual, type='yaw')
        pitch_angle = self.class2angle(pitch_class, pitch_residual, type='pitch')
        obb = np.zeros((5,))
        obb[0:3] = center
        obb[3] = yaw_angle
        obb[4] = pitch_angle
        return obb

    def vector2Euler(self, vector):
        xd = vector[0]
        yd = vector[1]
        zd = vector[2]
        yaw = math.atan2(yd, xd)
        assert -math.pi <= yaw <= math.pi
        pitch = math.asin(zd)
        assert -math.pi / 2 <= pitch <= math.pi / 2
        return yaw, pitch

    def Euler2vector(self, yaw, pitch):
        xd = math.cos(pitch) * math.cos(yaw)
        yd = math.cos(pitch) * math.sin(yaw)
        zd = math.sin(pitch)
        return np.asarray([xd, yd, zd])