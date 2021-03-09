# Tensorflow 2 Object Detection API

    Bu tutorial, TensorFlow 2.x'in kararlı sürümü olan TensorFlow 2.3'ye yöneliktir.
    
Bu, görüntülerde / videoda nesne algılamayı gerçekleştirmek için TensorFlow’un Nesne Algılama API'sini kurmaya ve kullanmaya yönelik adım adım bir kılavuzdur.

Takip ettiğim rehbere [buradan](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) ulaşabilirsiniz.

Bu eğitim boyunca kullanacağımız yazılım araçları aşağıdaki tabloda listelenmiştir:
    
![alt text](https://i.ibb.co/TmqDLk4/1.jpg)

##  Anaconda Python 3.7 yükleyin


Sistem gereksinimlerinize göre Python 3.7 64-Bit Graphical Installer veya 32-Bit Graphical Installer yükleyicisini [indirin](https://www.anaconda.com/products/individual).

(İsteğe bağlı) Sonraki adımda, "Add Anaconda3 to my PATH environment variable” Bu, Anaconda'yı varsayılan Python dağıtımınız yapar ve tüm düzenleyiciler arasında aynı varsayılan Python dağıtımına sahip olmanızı sağlar.

## Yeni bir Anaconda sanal ortamı oluşturun

Yeni bir Terminal penceresi açın

Aşağıdaki komutu yazın:

    conda create -n tensorflow pip python=3.8
Yukarıdakiler, tensorflow adlı yeni bir sanal ortam oluşturacaktır.

## Sanal ortami aktifleştirme

Yeni oluşturulan sanal ortamın etkinleştirilmesi cmd ekranında aşağıdakilerin çalıştırılmasıyla sağlanır:

    conda activate tensorflow
Sanal ortamınızı etkinleştirdikten sonra, ortamın adı cmd yol belirleyicinizin başında parantez içinde görüntülenmelidir, örneğin:    
    
    (tensorflow) C:\Users\Asus>
    
## TensorFlow kurulumu
TensorFlow kurulumu, 3 basit adımda yapılabilir.
#### TensorFlow PIP ile yükleyin
cmd de alttaki kodu çalıştırın. 

    pip install --ignore-installed --upgrade tensorflow==2.3.0
    
#### Kurulumunuzu doğrulayın
Aşağıdaki komutu  Terminal penceresinde çalıştırın:
    
    python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
Çalıştırdıktan sonra çıktınız şu şekilde olmalıdır:
    
    2020-06-22 19:20:32.614181: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
    2020-06-22 19:20:32.620571: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    2020-06-22 19:20:35.027232: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
    ith strength 1 edge matrix:
    . 
    .
    .
    2020-06-22 19:20:35.196815: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]
    tf.Tensor(1620.5817, shape=(), dtype=float32)
## GPU desteği
TensorFlow'u çalıştırmak için bir GPU kullanmak gerekli olmasa da, hesaplama açısından önemli. Bu nedenle, bilgisayarınıza uyumlu bir CUDA etkin GPU ile donatılmışsa, TensorFlow'un GPU'nuzu kullanmasını sağlamak için gerekli olan ilgili dosyaları yüklemek için aşağıda listelenen adımları izlemeniz önerilir.

Varsayılan olarak, TensorFlow çalıştırıldığında uyumlu GPU cihazlarını kaydetmeye çalışır. Bu başarısız olursa, TensorFlow platformun CPU'sunda çalışmaya başvuracaktır. Bu aynı zamanda, eksik kitaplık dosyalarını bildiren bir hata verir. 

    Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
![alt text](https://raw.githubusercontent.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector/master/doc/cuda.png)

TensorFlow'un GPU'nuzda çalışması için aşağıdaki gereksinimlerin karşılanması gerekir:

![alt text](https://i.ibb.co/VBhgM8K/2.jpg)
#### Cuda kurulumu
CUDA Toolkit 10.1'i [buradan](https://developer.nvidia.com/cuda-10.1-download-archive-update2?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork) indirebilirsiniz.

İndirdikten sonra Ortam değişkenleri /Sistem değişkenleri /Path den pathlerinizi düzenlemeniz gerekebilir:

![alt text](https://i.ibb.co/Kbf45VB/3.jpg)
#### CUDNN kurulumu

Https://developer.nvidia.com/rdp/cudnn-download adresine gidin.

Gerekirse bir kullanıcı profili oluşturun ve oturum açın.

CUDA 10.1 için cuDNN v7.6.5'iseçin

Windows 10 için cuDNN v7.6.5 dosyasını indirin

zip dosyasını(cuda) klasörünü \ NVIDIA GPU Computing Toolkit \ CUDA \ v10.1 \ dizinine çıkarın.

![alt text](https://i.ibb.co/xzDDrsk/4.jpg)

Ortam değişkenlerine PATH ini ekleyin.
![alt text](https://i.ibb.co/pP336Pc/5.jpg)

### GPU desteğini doğrulama
Gpu desteğini doğrulamak için kodu çalıştırın:

    python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
Çıktınız şu şekilde olmalıdır: 
![alt text](https://i.ibb.co/0ys9TPC/6.jpg)

- [x] Anaconda kurulumu
- [x] Tensorflow kurulumu
- [x] Gpu desteği
- [ ] Object Detection API kurulmu

# TensorFlow Object Detection API Kurulumu 
