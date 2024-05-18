Yapay Sinir Ağı ile Geri Yayılım (Back Propagation)

- Bu Python kodu, yapay sinir ağlarını kullanarak geriye yayılım algoritmasını uygular. Geriye yayılım, yapay sinir ağlarının eğitilmesinde
    sıkça kullanılan bir yöntemdir. Bu kod, dosya dizininde bulunan ornek_veriseti1.xls adlı veri seti üzerinde model eğitimi gerçekleştirir ve her 
    çalıştırma aşamasında yeniden oluşturulan bias ağırlıklarına göre tahminler yapar. Araştırmalarım sonucunda bias ağırlıklarının sabit tutulması 
    modele bağlıdır fakat ben random olarak yeni oluşturmayı tercih ettim.Bu yüzden her çalıştırma aşamasında modelin doğruluğu tahmin etme oranı 
    değişmektedir. Model test verisi için elde edilen sonuçları terminal ekranından takip edebilirsiniz.
    
Hedef: 
- Bu model excel veri setinde bulunan hedef sütununu yani y değişkenine atanan değer yaş değişkenidir. Amacım ise elde edilen test verilerinin gerçek
dünyada karşılık gelen ve aynı cevapları veren kişilerin yaş aralığının tahmin etmek için modellemesidir.

Kullanılan Kütüphaneler
- numpy
- matplotlib
- pandas
- scipy


Proje Yapısı
- ornek_veriseti1.xlsx: Proje içerisinde kullanılan örnek veri seti Excel dosyası.
- Furkan_Atasert_Back_Propagation.py: Ana Python dosyası, yapay sinir ağını tanımlar, eğitir ve tahminler yapar.



Kullanım
- Proje dosyalarını indirin.
- ornek_veriseti1.xlsx dosyasını proje klasörüne ekleyin.
- Ana Python dosyasını çalıştırın: python Furkan_Atasert_Back_Propagation.py

Ana Kod Parçaları
- NeuralNetwork sınıfı: Yapay sinir ağını tanımlar.
- forward metodu: İleri besleme işlemini gerçekleştirir.
- backward metodu: Geriye yayılım işlemini gerçekleştirir.
- fit metodu: Modeli eğitir.
- predict metodu: Test verisi üzerinde tahmin yapar.


Parametreler
- epochs: Eğitim iterasyonları sayısı.
- learning_rate: Öğrenme oranı.

Notlar
- Bu kod, verilen örnek veri seti üzerinde eğitilmiş ve sonuçlar elde edilmiştir.
- Modelin doğruluğunu değerlendirmek için hata değerleri ve görselleştirmeler kullanılmıştır.