from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamg.core.config import cfg
from siamg.tracker.nano_tracker import siamger

TRACKS = {
          'siamger': siamger 
         } 

def build_tracker(model): 
    return TRACKS[cfg.TRACK.TYPE](model) 
