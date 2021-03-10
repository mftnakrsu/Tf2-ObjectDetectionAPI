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

## Sanal ortamı aktifleştirme

Yeni oluşturulan sanal ortamın etkinleştirilmesi cmd ekranında aşağıdakilerin çalıştırılmasıyla sağlanır:

    conda activate tensorflow
Sanal ortamınızı etkinleştirdikten sonra, ortamın adı cmd yol belirleyicinizin başında parantez içinde görüntülenmelidir, örneğin:    
    
    (tensorflow) C:\Users\Asus>
    
## TensorFlow kurulumu
TensorFlow kurulumu, 3 basit adımda yapılabilir.
#### TensorFlow PIP ile yükleyin
Anaconda Promp tensorflow sanal ortamında de alttaki kodu çalıştırın. 

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
Artık TensorFlow'u kurduğunuza göre, TensorFlow Object Detection API'sini kurmanın zamanı geldi.

## TensorFlow Model Garden yükleme

TensorFlow Object Detection API'si için, modelimizi eğitmek için izlememiz gereken belirli bir dizin düzeni vardır.

İlk olarak, doğrudan C: içinde bir klasör oluşturun ve "TensorFlow" olarak adlandırın. Klasörü nereye yerleştireceğiniz size kalmış, ancak takibi kolay olması açısından ben C diskinin içinde oluşturdum. Bu klasörü oluşturduktan sonra Anaconda Promt'a geri dönün.
        
    activate tensorflow
    cd C:\TensorFlow
Bu dizine geldiğinizde, TensorFlow modelleri reposunu klonlamanız gerekecek.

![alt text](https://i.ibb.co/bgWzCKY/7.jpg)

    git clone https://github.com/tensorflow/models.git
    
![alt text](https://i.ibb.co/Jpxfm83/8.jpg)
 
En son, dizin yapınız şuna benzer görünmelidir.
 
    TensorFlow/
    └─ models/
       ├─ community/
       ├─ official/
       ├─ orbit/
       ├─ research/
       └── ...
      
Dizin yapısını kurduktan sonra, Object Detection API için ön koşulları yüklemeliyiz. İlk önce protobuf derleyicisini Anaconda Promt'da indiriyoruz.
  
    (tensorflow) C:\TensorFlow>
    conda install -c anaconda protobuf
    
Daha sonra TensorFlow \ models \ research dizinine gidin ve protobuf derleyecisini çalıştırın.
   
    cd models\research
    protoc object_detection\protos\*.proto --python_out=.

NOT: Ortam değişkenlerindeki değişikliklerin etkili olması için yeni bir Terminal açmanız GEREKİR.

## COCO API kurulumu

TensorFlow 2.x itibariyle, pycotools paketi Object Detection API'sinin bir [destek dosyaları](https://github.com/tensorflow/models/blob/master/research/object_detection/packages/tf2/setup.py) olarak listelenmiştir. İdeal olarak, bu paket, daha sonra da kurulabilir ama bazı hatalar alınabilir olduğu için şimdi kuracağız.

Bunu yaptıktan sonra, terminali kapatın ve yeni bir Anaconda Prompt açın açın. *activate tensorflow* ile sanal ortamınızı aktifleştirin.

    pip install cython
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    
Burda hata alabilirsiniz:

Kurulum talimatlarına göre Visual C ++ 2015 derleme araçlarının yüklü ve sizin pathinizde olması gerektiğini unutmayın. Bu pakete sahip değilseniz, [buradan](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&amp;rel=16) indirin.

Bunu da kurduktan sonra

    cd C:\TensorFlow\models\research
dizinine gidin ve
    
    copy object_detection\packages\tf2\setup.py .
    python -m pip install .
object detection api kurulumun tamamlayın.
Herhangi bir hata alırsanız, bildirin lütfen ancak bunlar büyük olasılıkla yüklemenizin yanlış olduğu anlamına gelen pycotools sorunlarıdır. Ancak her şey plana göre giderse kurulumunuzu test edebilirsiniz.

Kurulumu test etmek için *Tensorflow \ models \ research* içinden aşağıdaki komutu çalıştırın:

    python object_detection/builders/model_builder_tf2_test.py
Yukarıdakiler çalıştırıldığında, testin tamamlanması için biraz zaman tanıyın ve bittiğinde kurulumlarda hata yoksa aşağıdakine benzer bir çıktı almalısınız:    

    ...
    [       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
    [ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
    [       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
    [ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
    [       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
    [ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
    [       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
    [ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
    [       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
    [ RUN      ] ModelBuilderTF2Test.test_session
    [  SKIPPED ] ModelBuilderTF2Test.test_session
    [ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
    [       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
    [ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
    [       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
    [ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
    [       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
    ----------------------------------------------------------------------
    Ran 20 tests in 73.510s

    OK (skipped=1)
Bu, Anaconda Dizin Yapısını ve Object Detection API'sini başarıyla kurduğumuz anlamına gelir. Artık veri setimizi toplayıp kendi custom modelimizi oluşturabiliriz. Öyleyse bir sonraki adıma geçelim!

To do:
- [x] Object Detection API kurulmu
- [ ] Training Custom Object Detector

## Training Custom Object Detector

Burada kendi nesne dedektörünüzü nasıl eğitebileceğinizi göreceğiz.

1. Çalışma alanınızı / eğitim dosyalarınızı nasıl düzenleyebilirsiniz?  
2. Görüntü veri kümeleri nasıl hazırlanır / labellanır ?  
3. Bu tür veri kümelerinden tf record dosyaları nasıl oluşturulur?  
4. Basit bir pipeline nasıl konfigür edilir ?   
5. Bir model nasıl eğitilir ve ilerlemesi nasıl izlenir?  
6. Elde edilen model nasıl export edilir ve nesneleri algılamak için kullanılır?  

### Workspace hazırlama

1. Şu anda <PATH_TO_TF> altına yerleştirilmiş bir Tensorflow klasörünüz olmalıdır (örn. C: /TensorFlow), aşağıdaki path ağacı gibi: 

        TensorFlow/
        ├─ addons/ (Optional)
        │  └─ labelImg/
        └─ models/
           ├─ community/
           ├─ official/
           ├─ orbit/
           ├─ research/
           └─ ...
       
2. Şimdi TensorFlow altında yeni bir klasör oluşturun ve bunu 'workspace' olarak adlandırın. İstediğiniz ismi verebilirsini ama takibi kolay olsun diye aynı yaparsanız daha iyi olur. 'workspace' tüm train kurulumlarımızın olduğu dosya olacak. Şimdi çalışma alanının altına geçelim ve training_demo adlı başka bir klasör oluşturalım. Şimdi dizin yapımız şu şekilde olmalıdır:

        TensorFlow/
        ├─ addons/ (Optional)
        │  └─ labelImg/
        ├─ models/
        │  ├─ community/
        │  ├─ official/
        │  ├─ orbit/
        │  ├─ research/
        │  └─ ...
        └─ workspace/
           └─ training_demo/
           
3. Training_demo klasörü, model eğitimimizle ilgili tüm dosyaları içeren eğitim klasörümüz olacaktır. Farklı bir veri kümesi üzerinde eğitim almak istediğimiz her seferde ayrı bir eğitim klasörü oluşturmanız tavsiye edilir. Eğitim klasörlerinin tipik yapısı aşağıda gösterilmiştir.

        training_demo/
        ├─ annotations/
        ├─ exported-models/
        ├─ images/
        │  ├─ test/
        │  └─ train/
        ├─ models/
        ├─ pre-trained-models/
        └─ README.md
        
Yukarıdaki ağaçta gösterilen klasörlerin / dosyaların her biri için bir açıklama:
![alt text](https://i.ibb.co/pfjwDzf/10.jpg)


### Dataset hazırlama

Bir modeli kendi özel veri kümenizde eğitmek istiyorsanız, önce görüntüleri toplamalısınız. İdeal olarak her class için 100 resim kullanabilirsiniz. Örneğin, bir kedi ve köpek  detektörü eğitiyorsunuz. 100 kedi resmi ve 100 köpek resmi toplamanız gerekir. Kendi veri kümeniz için, farklı arka planlara ve açılara sahip çeşitli fotoğraflar çekmenizi tavsiye ederim.

![alt text](https://www.ilgitrafik.com/wp-content/uploads/2019/12/azami-h%C4%B1z-s%C4%B1n%C4%B1rlamas%C4%B1-30-km-levhas%C4%B1-tt-29-trafik-tanzim-i%C5%9Faretleri-levhalar%C4%B1-nedir-trafik-tanzim-levhalar%C4%B1-anlamlar%C4%B1-tanzim-levhas%C4%B1-fiyat%C4%B1-imalat%C4%B1-%C3%BCretimi-ankara-tanzim-i%C5%9Faret-levhas%C4%B1.jpg)

![alt text](https://foto.sondakika.com/haber/2014/11/21/gorme-engellilerin-yoluna-dikilen-dur-tabelasi-6708058_2498_m.jpg)

Verileri topladıktan sonra, veri kümesini ayırmalısınız. Bununla, verileri bir train seti ve test/valide setine ayırmanız gerekir.. Resimlerinizin % 80'ini images \ training klasörüne ve kalan % 20'sini images \ test klasörüne koymalısınız. Resimlerinizi ayırdıktan sonra, onları [LabelImg](https://github.com/tzutalin/labelImg) ile etiketleyebilirsiniz.

LablelImg'ı indirdikten sonra, Open Dir ve Save Dir gibi ayarları yapın. Bu, tüm görüntülerde dolaşmanıza ve nesnelerin etrafında bounding box ve etiketler oluşturmanıza yarar. Resminizi etiketledikten sonra kaydettiğinizden ve sonraki resme geçtiğinizden emin olun. Bunu images \ test and images \ train klasörlerindeki tüm görüntüler için yapın.

![alt text](https://i.ibb.co/LxGVDtS/11.jpg)
![alt text](https://i.ibb.co/zbvSQW6/12.jpg)

#### Open Dir  ve Save Dir'i de hallettikten sonra resimlerinizi labellamaya başlayabilirsiniz. Bence en eğlenceli kısmı burası.(!)

![alt text](https://i.ibb.co/4mkb4Tf/13.jpg)

## Label Map oluşturulması

TensorFlow, kullanılan etiketlerin her birini bir tam sayı değeriyle eşler. Bu label map hem eğitim hem de tespit süreçleri tarafından kullanılır.

Aşağıda, veri setimizin 2 etiket, 'dur' ve 'hiz30' içerdiğini varsayarak örnek bir etiket haritası (ör. Label_map.pbtxt) gösteriyoruz:

    item {
        id: 1
        name: 'dur'
    }

    item {
        id: 2
        name: 'hiz30'
    }
    
Örneğin, bir kedi, köpek ve iguana dedektörü yapmak istiyorsanız, etiket haritanız şunun gibi görünecektir:
    
    item {
        id: 1
        name: 'kedi'
    }

    item {
        id: 2
        name: 'köpek'
    }
    
     item {
        id: 3
        name: 'iguana'
    }
