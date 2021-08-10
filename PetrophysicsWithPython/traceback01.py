import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lasio

# Specify file path
filepath = 'https://raw.githubusercontent.com/yohanesnuwara/formation-evaluation/main/data/volve/15_9-F-11A.LAS'

# Read with lasio
las = lasio.read(filepath)
