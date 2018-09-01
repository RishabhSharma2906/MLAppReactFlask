import pandas as pd
import numpy as np
import functions as pf
import model as md

df = pd.read_csv('data/train.tsv',sep='\t', header=0)

print(pf.process())

val = md.model()

def predict(statement):
	return val
