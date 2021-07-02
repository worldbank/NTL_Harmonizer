import pandas as pd
import numpy as np
from pathlib import Path
import os

GDPFPATH= Path(os.environ['NLT'], "NTL_Harmonizer","files","nuts3_gdp_EU.csv")

def convert_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def load_prep(srcpath=GDPFPATH):
    df = pd.read_csv(srcpath, sep=';',engine="python",skipfooter=3)
    return df.set_index('GEO/TIME').T.applymap(lambda x: convert_float(x))
