import pandas as pd
import matplotlib.pyplot as plt

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")
gender_submission_df = pd.read_csv("gender_submission.csv")

categorical_columns = ['Sex', 'Embarked', 'Pclass']

def memory_usage(df):
    return df.memory_usage(deep=True).sum() / 1024

def convert_to_categorical(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

initial_memory = memory_usage(train_df)

convert_to_categorical(train_df, categorical_columns)
convert_to_categorical(test_df, categorical_columns)

final_memory = memory_usage(train_df)

print(f"Пам'ять до перетворення: {initial_memory:.2f} KB")
print(f"Пам'ять після перетворення: {final_memory:.2f} KB")
print(f"Економія пам'яті: {initial_memory - final_memory:.2f} KB")

print("\nТипи даних до перетворення:")
print(train_df.dtypes)

print("\nКількість унікальних значень у категоріальних стовпцях:")
for col in categorical_columns:
    if col in train_df.columns:
        print(f"Унікальних значень у стовпці {col}: {train_df[col].nunique()}")

labels = ['Before', 'After']
memory_usage_data = [initial_memory, final_memory]

plt.figure(figsize=(6, 4))
plt.bar(labels, memory_usage_data, color=['red', 'green'])
plt.title('Пам\'ять до та після перетворення категоріальних стовпців')
plt.ylabel('Пам\'ять (KB)')
plt.show()

unique_values = [train_df[col].nunique() for col in categorical_columns if col in train_df.columns]

plt.figure(figsize=(6, 4))
plt.bar(categorical_columns, unique_values, color='blue')
plt.title('Кількість унікальних значень у категоріальних стовпцях')
plt.ylabel('Кількість унікальних значень')
plt.xticks(rotation=45)
plt.show()
