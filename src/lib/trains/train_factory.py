from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mot import MotTrainer
from .mot_multiknob import MotTrainer_MultiKnob

train_factory = {
  'mot': MotTrainer,
  'mot_multiknob': MotTrainer_MultiKnob
}
