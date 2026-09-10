import matplotlib.pyplot as plt

# A generating predicted probabilities for the test data using the trained model
proba = cls.predict_proba(Xt)  # A
plt.figure(figsize=(8, 6))

# B creating a histogram of the predicted probabilities with specified bins and normalized density
plt.hist(proba, bins=30, density=True, color='blue', alpha=0.7)  # B
plt.xlabel('Predicted Probabilities')
plt.ylabel('Density')
plt.title('Histogram of Predicted Probabilities')
plt.grid(True)
plt.show()


