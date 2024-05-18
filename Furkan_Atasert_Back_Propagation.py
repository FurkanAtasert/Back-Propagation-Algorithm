######## HIZLI İŞLEM SAĞLAMANIZ ADINA kütüphane import alanı########

# pip install numpy matplotlib pandas scipy openpyxl

######## HIZLI İŞLEM SAĞLAMANIZ ADINA proje çalıştırma alanı ########

# python Furkan_Atasert_Back_Propagation.py

# Proje içeriğini dosya dizininde bulunan README.txt kısmında anlattım


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  
from scipy import stats


# Veri setini yükle
data = pd.read_excel("simple-dataset.xlsx", sheet_name="Clear_Data")

# Sigmoid aktivasyon fonksiyonu
def relu(x):
    return np.maximum(0, x)

# Sigmoid fonksiyonunun türevi
def relu_derivative(x):
    return np.where(x <= 0, 0, 1)

# Yapay Sinir Ağı sınıfı
class NeuralNetwork:
    def __init__(self, layers):
        # Sinir ağı katmanlarının listesi
        self.layers = layers
        # Ağırlıkların rastgele başlatılması
        self.weights = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]) for i in range(len(layers) - 1)]
        # Bias'ların rastgele başlatılması
        self.biases = [np.zeros((1, layers[i+1])) for i in range(len(layers) - 1)]
    
    # İleri Besleme Fonksiyonu
    def forward(self, x):
        # Z değerlerinin ve aktivasyon değerlerinin listeleri
        self.z_values = []
        self.activation_values = [x]
        for i in range(len(self.weights)):
            # Aktivasyon fonksiyonunun çıktısının hesaplanması
            z = np.dot(self.activation_values[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            activation = relu(z)
            self.activation_values.append(activation)
        return self.activation_values[-1]
    
    # Geriye Yayılım Fonksiyonu
    def backward(self, y):
        # Hata hesaplanması
        self.error = y - self.activation_values[-1]
        # Hatanın delta değerinin hesaplanması
        self.delta = [self.error]
        for i in range(len(self.z_values) - 1, 0, -1):
            error = np.dot(self.delta[-1], self.weights[i].T)
            delta = error * relu_derivative(self.activation_values[i])
            self.delta.append(delta)
        self.delta.reverse()
        
        # Gradientlerin hesaplanması
        self.gradient_weights = []
        self.gradient_biases = []
        for i in range(len(self.weights)):
            gradient_weight = np.dot(self.activation_values[i].T, self.delta[i])
            gradient_bias = np.sum(self.delta[i], axis=0)
            self.gradient_weights.append(gradient_weight)
            self.gradient_biases.append(gradient_bias)
        
    # Ağırlıkların ve biasların güncellenmesi
    def update_weights(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * self.gradient_weights[i]
            self.biases[i] += learning_rate * self.gradient_biases[i]
    
    # Modelin eğitilmesi
    def fit(self, X_train, y_train, epochs, learning_rate):
        self.loss = []
        for epoch in range(epochs):
            # İleri besleme, geriye yayılım ve güncelleme işlemleri
            self.forward(X_train)
            self.backward(y_train)
            self.update_weights(learning_rate)
            # Hata hesaplanması
            loss = np.mean(np.square(self.error))
            self.loss.append(loss)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
        print("Eğitim tamamlandı.")
    
        
    # Test verisi üzerinde tahmin yapma
    def predict(self, X_test):
        return self.forward(X_test)

# Veri setinin yüklenmesi ve girdi ve hedef verilerin ayrılması
X = data.drop(columns=["Age"]).values
y = data["Age"].values.reshape(-1, 1)

# Model oluşturma ve eğitme (GİZLİ KATMAN OLUŞTURMA) 
nn = NeuralNetwork([X.shape[1], 16, 8, 4, 1])
nn.fit(X, y, epochs=1000, learning_rate=0.0001)

# Tahmin yapma
predictions = nn.predict(X)
rounded_predictions = np.round(predictions)

# Tahmin edilen değerlerin gösterilmesi
print("Tahmin Edilen Değerler:")
for i in range(len(X)):
    print(f"Girdi: {X[i]}, Gerçek Değer: {y[i]}, Tahmin: {rounded_predictions[i]}")
    
# Doğruluk hesaplama
accuracy = np.mean(rounded_predictions == y)
print(f"Modelin tahmini %{accuracy*100:.2f} doğruluk oranına sahiptir. Bu, modelin eğitim veri setindeki girdilere dayanarak hedef değerleri %{accuracy*100:.2f} oranı doğrultusunda doğru bir şekilde tahmin etme başarısı gösteriyor. Bu oran ne kadar yüksekse, modelin veriler üzerindeki performansı o kadar iyidir.\nDeveloper Name: Furkan Atasert")

# Hataların hesaplanması
residuals = y - predictions
print("Hataların Boyutu:", residuals.shape)
residuals = residuals.flatten()


# Loss değerlerinin grafiği
plt.figure(figsize=(10, 5))
plt.plot(nn.loss)
plt.title("Loss Değerlerinin Değişimi")
plt.xlabel("Epochs")
plt.ylabel("Kayıp (Loss)")
plt.grid(True)
plt.text(0.5, 0.95, "Modelin eğitim sürecindeki loss değerlerinin değişimi", ha='center', va='center', transform=plt.gca().transAxes)
plt.show()

# Hataların histogramı
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title("Model Hatalarının Frekans Histogramı")
plt.xlabel("Hata Değeri")
plt.ylabel("Frekans")
plt.grid(True)
plt.text(0.5, 0.95, "Modelin eğitim verisine göre hata frekansı histogramı", ha='center', va='center', transform=plt.gca().transAxes)
plt.show()

# Q-Q plot grafiği
plt.figure(figsize=(10, 5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Grafiği")
plt.xlabel("Teorik Normal Değerler")
plt.ylabel("Sıralı Hatalar")
plt.grid(True)
plt.text(0.5, 0.95, "Modelin tahminlerinin gerçek verilere uyumu", ha='center', va='center', transform=plt.gca().transAxes)
plt.show()
