# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""List of subsets."""

DATASETS = {
    'baseball_pitch': {'train': 104, 'val': 63},
    'baseball_swing': {'train': 115, 'val': 58},
    'bench_press': {'train': 69, 'val': 71},
    'bowl': {'train': 135, 'val': 85},
    'clean_and_jerk': {'train': 43, 'val': 45},
    'golf_swing': {'train': 89, 'val': 77},
    'jumping_jacks': {'train': 56, 'val': 56},
    'pushup': {'train': 104, 'val': 107},
    'pullup': {'train': 98, 'val': 101},
    'situp': {'train': 50, 'val': 50},
    'squat': {'train': 114, 'val': 117},
    'tennis_forehand': {'train': 80, 'val': 77},
    'tennis_serve': {'train': 115, 'val': 71},
    'jump_rope': {'train': 40, 'val': 42},
    'strum_guitar':{'train': 46, 'val': 48},
}


DATASET_TO_NUM_CLASSES = {
    'baseball_pitch': 4,
    'baseball_swing': 3,
    'bench_press': 2,
    'bowling': 3,
    'clean_and_jerk': 6,
    'golf_swing': 3,
    'jumping_jacks': 4,
    'pushups': 2,
    'pullups': 2,
    'situp': 2,
    'squats': 4,
    'tennis_forehand': 3,
    'tennis_serve': 4,
}
