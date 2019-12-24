

import os

import subprocess

def say_notify(contents):
    cmd = "say" + contents
    # returns output as byte string
    returned_output = subprocess.check_output(cmd)
    print('Current date is:', returned_output.decode("utf-8"))
