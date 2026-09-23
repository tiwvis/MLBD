import pandas as pd

# Sample dataset of house prices by city
data = {
    'City': ['New York', 'Paris', 'New York', 'Tokyo', 'Paris', 'Tokyo', 'New York', 'Paris'],
    'Price': [500, 400, 600, 300, 450, 350, 550, 410]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Calculate the target mean per category and assign it to a new column
df['City_Encoded'] = df.groupby('City')['Price'].transform('mean')

print("\nEncoded DataFrame:")
print(df)