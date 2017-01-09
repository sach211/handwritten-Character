# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 00:58:31 2016

@author: Sachi Angle
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

letter_data = pd.read_csv('letter-recognition.csv', na_values = ['NA'])
sb.pairplot(letter_data.dropna(),hue='lttr')

for column_index, column in enumerate(letter_data.columns):
    if column == 'lttr':
        continue
    print(column)
    #plt.subplot(2, 2, column_index + 1)
    sb.violinplot(x='lttr', y=column, data=letter_data)
    plt.show()
    
    


