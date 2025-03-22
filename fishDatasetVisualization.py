#A program to visualize the fish species dataset using scatter plots and pie charts.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Taking file path as input from the user.
filePath = input("Enter the file path of the fish species dataset (CSV): ")
data = pd.read_csv(filePath)

#Calculating Volume (Length1 × Length2 × Length3), and storing as a new column.
data["Volume"] = data["Length1"] * data["Length2"] * data["Length3"]

#Set figure size for better visualisation.
plt.figure(figsize=(12, 8))

#A Scatter plot: Species vs Weight.
plt.subplot(2, 2, 1)
sns.scatterplot(x=data["Species"], y=data["Weight"], hue=data["Species"], palette="Set2", alpha=0.9, s=100, edgecolor="black")
plt.xticks(rotation=45)
plt.title("Species vs Weight")
plt.xlabel("Species")
plt.ylabel("Weight (g)")

#A Scatter plot: Species vs Volume.
plt.subplot(2, 2, 2)
sns.scatterplot(x=data["Species"], y=data["Volume"], hue=data["Species"], palette="Set2", alpha=0.9, s=100, edgecolor="black")
plt.xticks(rotation=45)
plt.title("Species vs Volume")
plt.xlabel("Species")
plt.ylabel("Volume (cm³)")

#A Scatter plot: Weight vs Volume (grouped by species).
plt.subplot(2, 2, 3)
sns.scatterplot(x=data["Volume"], y=data["Weight"], hue=data["Species"], palette="tab20", alpha=0.7, s=100, edgecolor="black")
plt.title("Weight vs Volume (Grouped by Species)")
plt.xlabel("Volume (cm³)")
plt.ylabel("Weight (g)")

#A Pie chart of Species distributions in the dataset.
plt.subplot(2, 2, 4)
speciesCounts = data["Species"].value_counts()
plt.pie(speciesCounts, labels=speciesCounts.index, autopct="%1.1f%%", colors=sns.color_palette("Set3"))
plt.title("Species Distribution")

#Displaying the plot.
plt.tight_layout()
plt.show()
