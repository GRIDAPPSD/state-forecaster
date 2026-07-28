#!/usr/bin/python3
"""
Created on July 28, 2026

@author: Gary Black
"""""

import sys
import time
import os
import json
import importlib
import pprint

from gridappsd import GridAPPSD
from gridappsd.topics import service_output_topic


class SFWrapper(object):
  def __init__(self, gapps):
    self.gapps = gapps
    self.keepLoopingFlag = True


  def keepLooping(self):
    return self.keepLoopingFlag


  def on_message(self, header, message):
    # TODO workaround for broken unsubscribe method
    if not self.keepLoopingFlag:
      return

    if 'processStatus' in message:
      status = message['processStatus']
      if status=='COMPLETE' or status=='CLOSED':
        print('State-Forecaster sent COMPLETE status message', flush=True)
        self.keepLoopingFlag = False

    else:
      print('State-Forecaster message: ' + str(message), flush=True)


class TestSF(GridAPPSD):
  def __init__(self, gapps, simulation_id):
    gapps_sim = GridAPPSD()

    self.sfRap = SFWrapper(gapps_sim)

    print('Subscribing to service output topic: ' + service_output_topic('state-forecaster', simulation_id) + '\n', flush=True)
    out_id = gapps_sim.subscribe(service_output_topic('state-forecaster', simulation_id), self.sfRap)

    print('Starting state-forecaster monitoring loop...\n', flush=True)

    while self.sfRap.keepLooping():
      #print('Sleeping...', flush=True)
      time.sleep(0.1)

    print('Finished state-forecaster monitoring loop.\n', flush=True)

    gapps_sim.unsubscribe(out_id)

    return


def _main():
  # for loading modules (this works for finding static-ybus too)
  if (os.path.isdir('shared')):
    sys.path.append('.')
  elif (os.path.isdir('../shared')):
    sys.path.append('..')
  elif (os.path.isdir('gridappsd-toolbox/shared')):
    sys.path.append('gridappsd-toolbox')
  else:
    sys.path.append('/gridappsd/services/gridappsd-toolbox')
   
  simulation_id = sys.argv[1]

  # authenticate with GridAPPS-D Platform
  os.environ['GRIDAPPSD_APPLICATION_ID'] = 'gridappsd-dynamic-ybus-service'
  os.environ['GRIDAPPSD_APPLICATION_STATUS'] = 'STARTED'
  os.environ['GRIDAPPSD_USER'] = 'app_user'
  os.environ['GRIDAPPSD_PASSWORD'] = '1234App'

  gapps = GridAPPSD(simulation_id)
  assert gapps.connected

  test_sim = TestSF(gapps, simulation_id)


if __name__ == "__main__":
  _main()

