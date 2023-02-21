from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset
from .dataset.jde import JointDataset_MultiKnob

def get_dataset(dataset, task):
  if task == 'mot':
    return JointDataset
  if task == 'mot_multiknob':
    return JointDataset_MultiKnob
  else:
    return None
  
