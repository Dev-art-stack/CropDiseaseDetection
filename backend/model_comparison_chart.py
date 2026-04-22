# Model comparison chart
import matplotlib.pyplot as plt
models  = ['MobileNetV3\n(Ours)', 'EfficientNet-B0', 'ResNet50', 'VGG16']
accs    = [98.03, 99.27, 99.59, 98.67]
colors  = ['green', 'steelblue', 'steelblue', 'steelblue']

plt.figure(figsize=(8,5))
plt.bar(models, accs, color=colors)
plt.ylim(96, 100)
plt.ylabel('Test Accuracy (%)')
plt.title('Model Comparison')
plt.savefig('model_comparison.pdf')