import warnings
warnings.filterwarnings("ignore") 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\Varshitha\Downloads\food_orders_new_delhi.csv")
print(f"\n Shape of dataset: \n",df.shape)
print(f"\n First 5 rows of a dataset: \n",df.head())
print(f"\n Datatypes used in dataset: \n",df.info())
print(f"\n Statistical Summary of the dataset: \n",df.describe())
print(f"\n No.of NULL values: \n",df.isnull().sum())
print(f"\n Total Missing values in dataset: \n",df.isnull().sum().sum())
print(f"\n updated missing values: ",df.fillna(df['Discounts and Offers'].mode()[0],inplace=True))
print(df.isnull().sum())


colors = ['#191970', '#36454F', '#50C878', '#8A0303', '#9966CC',
                '#008080', '#2a3439', '#6F00FF', '#00FFFF', '#9E5E6F']

#1.Bar Plot:Order by Day of Week
df['Delivery Date and Time'] = pd.to_datetime(df['Delivery Date and Time'], errors='coerce')
df['day_of_week'] = df['Delivery Date and Time'].dt.day_name()
plt.figure(figsize=(10, 6))
order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
sns.countplot(data=df, x='day_of_week', order=order, hue="day_of_week")
plt.title("Orders by Day of the Week", fontsize=14)
plt.xlabel("Day of the Week")
plt.ylabel("Orders")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Pie Chart: Payment Method Distribution
plt.figure(figsize=(6,6))
payment_counts = df['Payment Method'].value_counts()
plt.pie(payment_counts, labels=payment_counts.index, autopct='%1.1f%%',
        colors=colors, startangle=140)
plt.title("Payment Method Distribution")
plt.axis('equal')
plt.show()

# 3. Histplot: Distribution of Order Values
plt.figure(figsize=(8,5))
sns.histplot(df['Order Value'], bins=30, kde=True, color='#9E5E6F')
plt.title("Distribution of Order Values")
plt.xlabel("Order Amount")
plt.ylabel("Frequency")
plt.show()

# 4. Scatterplot: Order Value vs Delivery Fee
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Order Value', y='Delivery Fee', hue='Payment Method', alpha=0.8,palette='Dark2', edgecolor='black')
plt.title("Order Value vs Delivery Fee")
plt.xlabel("Order Amount")
plt.ylabel("Delivery Fee")
plt.legend(title="Payment Method")
plt.show()

# 5. Donut Chart: Refunds or Chargebacks
refund_counts = df['Refunds/Chargebacks'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(refund_counts, labels=refund_counts.index, startangle=90,
        colors=colors, wedgeprops=dict(width=0.3), autopct='%1.1f%%')
plt.title("Refunds or Chargebacks")
plt.axis('equal')
plt.show()

# 6. Line Plot: Average Order Value Over Time
print(df.columns.tolist())
daily_avg = df.groupby('Order Date and Time')['Order Value'].mean().reset_index()
plt.figure(figsize=(10,5))
sns.lineplot(data=daily_avg, x='Order Date and Time', y='Order Value', marker='o', color='b')
plt.title("Average Order Value Over Time")
plt.xlabel("Date")
plt.ylabel("Average Order Amount")
plt.show()

# 7. Box Plot: Order Value by Payment Method
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='Payment Method', y='Order Value',palette="plasma")
plt.title("Order Value by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Order Amount")
plt.xticks(rotation=45)
plt.show()

# 8. Heatmap: Correlation Matrix
plt.figure(figsize=(8,5))
corr_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(corr_matrix, annot=True, linewidths=0.5,cmap="cividis")
plt.title("Correlation Heatmap")
plt.show()
