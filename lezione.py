import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
iris = sns.load_dataset("iris")

# Display the first few rows
print(iris.head())

# 1. Pairplot: Scatter plot matrix with hue differentiation for species
sns.pairplot(iris, hue="species", height=2)
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# 2. Heatmap: Correlation matrix of numerical features
correlation = iris.drop("species", axis=1).corr()
plt.figure(figsize=(6, 4))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Iris Features")
plt.show()

# 3. Boxplot: Distribution of each feature by species
plt.figure(figsize=(10, 6))
iris_melted = iris.melt(id_vars="species")
sns.boxplot(x="variable", y="value", hue="species", data=iris_melted)
plt.title("Boxplot of Iris Features by Species")
plt.show()

# 4. Violin plot: Detailed distribution of features for each species
plt.figure(figsize=(10, 6))
sns.violinplot(x="variable", y="value", hue="species", data=iris_melted, split=True)
plt.title("Violin Plot of Iris Features by Species")
plt.show()