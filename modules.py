
import matplotlib.pyplot as plt

def plot_box(series, title, xlabel):
    cleaned_series = series.dropna()
    plt.figure(figsize=(8, 6))
    plt.boxplot(cleaned_series,
                vert=False)
    plt.xlabel(xlabel)
    plt.title(title)
