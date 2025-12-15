import pandas as pd

# Load Excel
file_path = r'C:\Users\griff\OneDrive\Desktop\SeedlingDataProject\data\raw\Organized Seedling Data.xlsx'
df = pd.read_excel(file_path)

# Clean
df['MeasurementType'] = df['MeasurementType'].str.strip()
df['MeasurementValue'] = pd.to_numeric(df['MeasurementValue'], errors='coerce')
df['Browsed'] = df['Browsed'].fillna('Unknown')
df['Coordinates'] = df['Coordinates'].fillna('Unknown')

# Add unique row ID
df['RowID'] = df.index

# Pivot only using RowID
df_wide = df.pivot_table(
    index='RowID',
    columns='MeasurementType',
    values='MeasurementValue',
    aggfunc='first'
).reset_index()

# Bring back original metadata
meta_cols = ['Date', 'Location', 'Species', 'Browsed', 'Coordinates', 'Notes', 'CanopyScore']
df_meta = df[meta_cols + ['RowID']]
df_final = pd.merge(df_wide, df_meta, on='RowID', how='left')

# Save
df_final.to_excel(r'C:\Users\griff\OneDrive\Desktop\SeedlingDataProject\wide_seedling_data.xlsx', index=False)
print(f"Rows: {len(df_final)}, Columns: {len(df_final.columns)}")