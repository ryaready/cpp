import numpy as np
import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])
cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])


cities['saint'] = cities['City name'].str.startswith('San')
cities['square'] = np.where(cities['Area square miles'] > 50, True, False)

cities['ex1'] =  cities['saint'] & cities['square']


cities = cities.drop(columns=['saint', 'square'])