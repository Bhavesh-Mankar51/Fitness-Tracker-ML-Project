import pandas as pd

# Load both CSV files
df1 = pd.read_csv('data/external/yourteam_file.csv')
df2 = pd.read_csv('data/external/yourteambench_file.csv')

# Check if they're identical
are_identical = df1.equals(df2)
print(f"Files are identical: {are_identical}")

# If they're not identical, you might want to examine the differences
if not are_identical:
    # Check if they have the same shape
    if df1.shape != df2.shape:
        print(f"Shapes differ: df1 is {df1.shape}, df2 is {df2.shape}")
    
    # Check for column differences
    cols_df1_not_in_df2 = set(df1.columns) - set(df2.columns)
    cols_df2_not_in_df1 = set(df2.columns) - set(df1.columns)
    
    if cols_df1_not_in_df2:
        print(f"Columns in df1 not in df2: {cols_df1_not_in_df2}")
    if cols_df2_not_in_df1:
        print(f"Columns in df2 not in df1: {cols_df2_not_in_df1}")
    
    # For columns in both dataframes, check for value differences
    common_cols = set(df1.columns) & set(df2.columns)
    for col in common_cols:
        if not df1[col].equals(df2[col]):
            print(f"Values differ in column: {col}")
            
            # Sample of differences
            mask = df1[col] != df2[col]
            if mask.any():
                sample_diff = pd.DataFrame({
                    'df1': df1.loc[mask, col].head(),
                    'df2': df2.loc[mask, col].head()
                })
                print(sample_diff)