"""
Otonom Yazılım Geliştirme Ekibi
CrewAI + Google Gemini 2.5 Flash

Mimari:
  Sistem Mimarı → Geliştirici → QA Uzmanı ⇄ Performans Mimarı
                                     ↑______________|
                               (geri bildirim döngüsü)
"""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# LLM — Google Gemini 2.5 Flash
# ─────────────────────────────────────────────────────────────────────────────
_api_key = os.environ.get("GOOGLE_API_KEY")
if not _api_key:
    raise EnvironmentError(
        "GOOGLE_API_KEY ortam değişkeni bulunamadı. "
        ".env.example dosyasını kopyalayıp .env olarak adlandırın "
        "ve geçerli API anahtarınızı girin."
    )

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=_api_key,
)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL — DockerExecutePythonTool
# ─────────────────────────────────────────────────────────────────────────────
@tool("DockerExecutePythonTool")
def docker_execute_python_tool(code: str) -> str:
    """
    Verilen Python kodunu izole bir Docker konteynerinde (python:3.10-slim) çalıştırır.
    Kod temp_script.py dosyasına yazılır, Docker ile yürütülür ve stdout ile stderr
    içeren sonuç metni döndürülür. İşlem 15 saniyeyi aşarsa timeout hatası verilir.
    Başarılı çalışma Return Code 0 ve boş STDERR anlamına gelir.
    """
    script_path = Path(os.getcwd()) / "temp_script.py"

    try:
        script_path.write_text(code, encoding="utf-8")

        # Windows'ta Docker Desktop, yolları forward slash ile bekler.
        if os.name == "nt":
            mount_src = str(script_path).replace("\\", "/")
        else:
            mount_src = str(script_path)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{mount_src}:/app/script.py",
            "--memory", "512m",
            "--cpus", "1",
            "python:3.10-slim",
            "python", "/app/script.py",
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
        )

        sections: list[str] = []
        if proc.stdout.strip():
            sections.append(f"STDOUT:\n{proc.stdout.strip()}")
        if proc.stderr.strip():
            sections.append(f"STDERR:\n{proc.stderr.strip()}")
        sections.append(f"Return Code: {proc.returncode}")

        return "\n\n".join(sections)

    except subprocess.TimeoutExpired:
        return "HATA: Yürütme 15 saniye sonra zaman aşımına uğradı (timeout)."
    except FileNotFoundError:
        return (
            "HATA: 'docker' komutu bulunamadı. "
            "Docker Desktop'ın kurulu ve çalışır durumda olduğundan emin olun."
        )
    except Exception as exc:
        return f"HATA: {type(exc).__name__}: {exc}"
    finally:
        if script_path.exists():
            script_path.unlink(missing_ok=True)


@tool("CodeWriterTool")
def code_writer_tool(code: str, filename: str) -> str:
    """
    Verilen Python kodunu, belirtilen dosya adıyla diske kalıcı bir .py dosyası olarak yazar.
    Dosya zaten varsa üzerine yazılır. Başarı durumunda dosyanın tam (absolute) yolunu döndürür.
    """
    if not filename.endswith(".py"):
        filename = f"{filename}.py"

    target_path = Path(os.getcwd()) / filename

    try:
        target_path.write_text(code, encoding="utf-8")
        return f"Kod başarıyla kaydedildi: {target_path.resolve()}"
    except Exception as exc:
        return f"HATA: Dosya yazılırken sorun oluştu: {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# AGENTS
# ─────────────────────────────────────────────────────────────────────────────
system_architect = Agent(
    role="Sistem Mimarı",
    goal=(
        "Yazılım projesini derinlemesine analiz et, kapsamlı bir teknik mimari plan "
        "oluştur ve Geliştirici için net, uygulanabilir alt görevler belirle."
    ),
    backstory=(
        "10 yıllık deneyime sahip kıdemli bir yazılım mimarısın. "
        "Karmaşık gereksinimleri küçük, yönetilebilir teknik görevlere ayırma konusunda uzmansın. "
        "Python standart kütüphaneleri ve dosya sistemi operasyonları hakkında derin bilgiye sahipsin. "
        "Planların daima açık, ölçülebilir ve doğrudan uygulanabilir olur; "
        "gereksiz kütüphane bağımlılıklarından özenle kaçınırsın."
    ),
    llm=gemini_llm,
    max_iter=10,
    verbose=True,
    allow_delegation=False,
)

developer = Agent(
    role="Geliştirici",
    goal=(
        "Mimari plana sadık kalarak temiz, test edilebilir ve verimli Python kodu üret. "
        "QA veya Performans Mimarından gelen geri bildirimleri hızla koda yansıt."
    ),
    backstory=(
        "Kıdemli bir Python geliştiricisisin. "
        "Verilen teknik planı alır, bunu PEP8 uyumlu, hata yönetimi içeren "
        "ve anlaşılır Python koduna dönüştürürsün. "
        "Eleştiriye açıksın ve geri bildirimleri hızla değerlendirip kodu revize edersin. "
        "Standart kütüphaneleri tercih eder, pathlib, shutil ve random modüllerine hakimsin. "
        "Gereksiz yorum satırları ve açıklama yazmak yerine, kendi kendini belgeleyen kod yazarsın."
    ),
    llm=gemini_llm,
    max_iter=10,
    verbose=True,
    allow_delegation=False,
    tools=[code_writer_tool],
)

qa_engineer = Agent(
    role="QA Uzmanı",
    goal=(
        "Yazılan kodu DockerExecutePythonTool aracıyla izole ortamda test et. "
        "Herhangi bir hata durumunda kodu Geliştiriciye iade et ve düzeltilmesini sağla."
    ),
    backstory=(
        "Titiz ve metodolojik bir kalite güvence mühendisisin. "
        "Hiçbir hatayı gözden kaçırmaz, kodu her zaman izole bir Docker ortamında test edersin. "
        "STDERR çıktısı veya sıfırdan farklı return kodu gördüğünde kodu reddeder, "
        "hatanın tam açıklamasıyla birlikte Geliştiriciye iade edersin. "
        "Kod kusursuz çalıştığında — ve yalnızca o zaman — onayı verirsin."
    ),
    llm=gemini_llm,
    max_iter=10,
    verbose=True,
    allow_delegation=True,
    tools=[docker_execute_python_tool],
)

performance_architect = Agent(
    role="Performans Mimarı",
    goal=(
        "QA onayından geçen kodu Big O karmaşıklığı, bellek yönetimi ve kaynak güvenliği "
        "açısından incele; gerekirse Geliştiriciye somut optimizasyon önerileri sun."
    ),
    backstory=(
        "Sistem performansı ve kaynak optimizasyonu konusunda uzman bir mimarsın. "
        "Algoritmaların zaman ve alan karmaşıklığını analiz eder, "
        "potansiyel bellek sızıntılarını tespit eder, "
        "dosya ve kaynak yönetiminin doğru yapıldığından emin olursun. "
        "with bloğu kullanımı, generator vs list tercihi, gereksiz döngüler "
        "ve magic number'lar gibi detaylara dikkat edersin. "
        "Önerilerini her zaman somut ve uygulanabilir biçimde sunar, "
        "gereksiz karmaşıklıktan kaçınırsın."
    ),
    llm=gemini_llm,
    max_iter=10,
    verbose=True,
    allow_delegation=True,
    tools=[code_writer_tool],
)


# ─────────────────────────────────────────────────────────────────────────────
# TASKS
# ─────────────────────────────────────────────────────────────────────────────
architecture_task = Task(
    description="""
Aşağıdaki hedef için kapsamlı bir teknik mimari plan hazırla.

HEDEF:
YOLO modeli eğitimi için bir kaynak klasördeki görüntü dosyalarını (.jpg, .jpeg, .png)
ve karşılık gelen etiket dosyalarını (.txt) alıp;
  - Train   → %70
  - Test    → %15
  - Validation → %15
oranında rastgele bölen, gerekli klasör hiyerarşisini oluşturan ve
dosya bütünlük kontrolü yapan bir Python betiği geliştirilmesi.

Planın şu başlıkları kapsamalı:
1. Kullanılacak Python standart kütüphaneleri ve her birinin amacı
2. Temel fonksiyonlar: imzaları, parametreleri ve dönüş değerleri
3. Hedef klasör yapısı şeması (train/images, train/labels, vb.)
4. Dosya bütünlük kontrolü mantığı (her görüntüye karşılık bir .txt olmalı)
5. Rastgele bölme algoritmasının çalışma prensibi
6. Hata senaryoları: eksik etiket, boş kaynak klasör, geçersiz uzantı
7. Geliştirici için numaralı, sıralı uygulama talimatları
""",
    expected_output=(
        "Geliştirici herhangi bir ek açıklama istemeksizin kodu yazabilecek kadar ayrıntılı "
        "teknik mimari belgesi. Kütüphane listesi, fonksiyon tasarımı (imzalar dahil), "
        "klasör yapısı şeması ve sıralı uygulama adımlarını içermeli."
    ),
    agent=system_architect,
)

development_task = Task(
    description="""
Sistem Mimarının planını kullanarak tam ve çalışır bir Python betiği yaz.

FONKSİYONEL GEREKSİNİMLER:
1. Kaynak klasörden tüm görüntü dosyalarını (.jpg, .jpeg, .png) listele
2. Her görüntünün aynı isimde bir .txt etiket dosyasının varlığını doğrula
   (eksik etiket → uyarı mesajı yaz ve o dosyayı atla)
3. Geçerli görüntü–etiket çiftlerini random.shuffle ile karıştır
4. %70 / %15 / %15 oranında böl (oranlar dosya başında sabit olarak tanımlanmalı)
5. Hedefte şu klasör yapısını oluştur:
     output/
       train/images/   train/labels/
       test/images/    test/labels/
       val/images/     val/labels/
6. Dosyaları shutil.copy2 ile kopyala (metadata korunsun)
7. Sonunda özet istatistik yazdır:
   toplam çift, atlanan, train/test/val sayıları

KOD KURALLARI:
- Yalnızca Python standart kütüphaneleri: os, shutil, random, pathlib, sys
- Tüm yolları pathlib.Path ile yönet
- Oranları dosyanın en üstünde sabit olarak tanımla: TRAIN_RATIO = 0.70 vb.
- Her fonksiyona tek satır docstring ekle
- Betik kendi içinde test senaryosu çalıştırmalı:
    * temp_dataset/ geçici klasörünü oluştur
    * 10 adet sahte .jpg + .txt çifti ve 2 etiketsiz .jpg yaz
      (gerçek resim gerekmez; boş veya tek satır metin içeren dosya yeterli)
    * Betiği bu geçici veri seti üzerinde çalıştır
    * Çıktıyı ekrana yazdır
    * temp_dataset/ ve output/ klasörlerini temizle (shutil.rmtree)

DOSYAYA KAYDETME:
Kodu yazdıktan sonra CodeWriterTool'u kullanarak bu koda uygun, açıklayıcı bir dosya adı seç
(örn. "dataset_splitter.py") ve kodu bu isimle diske kaydet. Görev çıktında hem tam Python
kodunu hem de CodeWriterTool'dan dönen kaydedilen dosya yolunu belirt.

ÇIKTI FORMATI: Python kodu ve ardından kaydedilen dosya yolu bilgisi. Markdown blok işareti
veya gereksiz açıklama YOK.
""",
    expected_output=(
        "Tek başına çalışabilen, kendi test ortamını kurup temizleyen tam Python kodu, "
        "ardından CodeWriterTool ile kaydedilen dosyanın tam yolu."
    ),
    agent=developer,
    context=[architecture_task],
)

qa_task = Task(
    description="""
Geliştiricinin yazdığı Python kodunu DockerExecutePythonTool aracıyla test et.

TEST PROTOKOLÜ:
1. Önceki görevden gelen Python kodunu DockerExecutePythonTool aracına gönder
2. Araç çıktısını analiz et:
   - STDERR boş VE Return Code 0  →  KOD ONAYLANDI
   - STDERR dolu VEYA Return Code ≠ 0  →  KOD REDDEDİLDİ

ONAY DURUMUNDA:
- "✅ KOD ONAYLANDI" başlığıyla test raporu oluştur
- STDOUT çıktısını ve başarı özetini ekle

RED DURUMUNDA:
- "❌ KOD REDDEDİLDİ" başlığıyla hata raporu oluştur
- STDERR metnini olduğu gibi yaz
- Geliştiriciye neyi, neden düzeltmesi gerektiğini açıkça belirt
- Görevi Geliştiriciye devret (delegate); düzeltilmiş kodu tekrar test et
- Kod onaylanana kadar döngüyü sürdür (max_iter sınırına kadar)
""",
    expected_output=(
        "Test raporu. '✅ KOD ONAYLANDI' ile başlayan başarı raporu "
        "veya '❌ KOD REDDEDİLDİ' ile başlayan hata analizi ve düzeltme talebi."
    ),
    agent=qa_engineer,
    context=[development_task],
)

performance_task = Task(
    description="""
QA onayından geçen kodu aşağıdaki dört kriter açısından incele
ve her biri için ✅ İyi / ⚠️ Orta / ❌ Kötü değerlendirmesi yap.

KRİTER 1 — ZAMAN KARMAŞIKLIĞI (Big O Analizi):
  - Dosya listeleme ve kopyalama döngülerinin karmaşıklığı nedir?
  - Gereksiz tekrarlı dosya sistemi traversal'ı var mı?
  - Liste ve sözlük operasyonlarının verimliliği yeterli mi?

KRİTER 2 — BELLEK YÖNETİMİ:
  - Büyük dosya listeleri nasıl tutulmuş (list vs generator)?
  - Gereksiz veri kopyalamaları veya ara veri yapıları var mı?
  - Yüz binlerce dosyalık veri setlerine ölçeklenebilir mi?

KRİTER 3 — KAYNAK GÜVENLİĞİ:
  - Dosya işlemleri için with bloğu kullanımı var mı?
  - shutil.copy2 tercihi doğru mu (metadata korunuyor mu)?
  - Açık kalabilecek herhangi bir kaynak var mı?

KRİTER 4 — KOD KALİTESİ:
  - Oranlar (0.70, 0.15) sabit olarak tanımlı mı, yoksa magic number mı?
  - Fonksiyonların tek sorumluluğu var mı (SRP)?
  - Hata mesajları yeterince açıklayıcı mı?

KARAR:
  - Bir veya daha fazla ❌ varsa → Geliştiriciye somut kod önerileriyle devret
  - Tüm kriterler ✅ veya ⚠️ ise → "✅ PERFORMANS ONAYI VERİLDİ" ile bitir

DOSYAYA KAYDETME:
Nihai onaylanmış/optimize kodu CodeWriterTool ile Geliştiricinin önceki adımda kaydettiği
dosyanın AYNI ADIYLA tekrar kaydet (üzerine yazılacak şekilde). Böylece diskteki dosya her
zaman en güncel, onaylanmış kodu içerir.

Raporun sonuna nihai, onaylanmış Python kodunu ve CodeWriterTool'dan dönen dosya yolunu ekle.
""",
    expected_output=(
        "Dört kriterli performans analiz raporu: her kriter için sembol ve gerekçe, "
        "varsa somut optimizasyon önerileri, raporun sonunda nihai onaylanmış Python kodu "
        "ve CodeWriterTool ile kaydedilen dosyanın tam yolu."
    ),
    agent=performance_architect,
    context=[qa_task],
)


# ─────────────────────────────────────────────────────────────────────────────
# CREW
# ─────────────────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[system_architect, developer, qa_engineer, performance_architect],
    tasks=[architecture_task, development_task, qa_task, performance_task],
    process=Process.sequential,
    verbose=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bar = "=" * 65
    print(f"\n{bar}")
    print("  OTONOM YAZILIM GELİŞTİRME EKİBİ")
    print("  CrewAI  ·  Google Gemini 2.5 Flash  ·  Docker Sandbox")
    print(f"{bar}\n")

    result = crew.kickoff()

    print(f"\n{bar}")
    print("  GÖREV TAMAMLANDI — NİHAİ ÇIKTI")
    print(f"{bar}\n")
    print(result)
