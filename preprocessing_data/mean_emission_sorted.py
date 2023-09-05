import pandas as pd
import numpy as np

df_original = pd.read_excel('raw_data/SuEatableLife_Food_Fooprint_database.xlsx', sheet_name='SEL CF DATA SOURCES')

selected_columns = ['FOOD COMMODITY GROUP', 'FOOD COMMODITY TYPOLOGY', 'Carbon footprint  (kg CO2 eq/kg or litre of food commodity)', 'Country', 'Region']
df_emission = df_original[selected_columns]

filtered_df = df_emission[df_emission['FOOD COMMODITY TYPOLOGY'] == 'BEER']
mean_emission = filtered_df['Carbon footprint  (kg CO2 eq/kg or litre of food commodity)'].mean()

mean_emission_sorted = (df_emission.groupby('FOOD COMMODITY TYPOLOGY')
                        .agg({'Carbon footprint  (kg CO2 eq/kg or litre of food commodity)': 'mean'}).reset_index()
                        .rename(columns={'Carbon footprint  (kg CO2 eq/kg or litre of food commodity)': 'Mean footprint (kg CO2 eq/kg or litre of commodity)'})
                        .sort_values(by='Mean footprint (kg CO2 eq/kg or litre of commodity)', ascending=False).reset_index(drop=True)
                        .round({'Mean footprint (kg CO2 eq/kg or litre of commodity)': 2})
                        )

file_path = 'mean_emission_sorted.csv'
mean_emission_sorted.to_csv(file_path, index=False)
