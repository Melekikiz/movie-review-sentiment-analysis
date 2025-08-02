from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

options = Options()
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")

driver = webdriver.Chrome(options=options)
driver.get("https://www.imdb.com/title/tt0111161/reviews/?ref_=tt_ururv_sm")

time.sleep(2)  # Sayfanın biraz yüklenmesini bekle

# Açık yorum kutusu varsa kapat
try:
    close_button = driver.find_element(By.XPATH, '//button[@aria-label="Close"]')
    driver.execute_script("arguments[0].click();", close_button)
    print("🔵 Açık katman kapatıldı.")
    time.sleep(2)
except:
    pass

body = driver.find_element(By.TAG_NAME, "body")

for i in range(120):
    # Sayfayı sonuna kaydır
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)  # Yorumların yüklenmesi için bekle

    # Eğer "25 more" butonu varsa tıkla
    try:
        load_more = WebDriverWait(driver, 3).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(., "25 more")]'))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
        time.sleep(1)
        load_more.click()
        print(f"✅ '25 more' butonuna tıklandı. Döngü: {i+1}")
        time.sleep(3)  # Yeni yorumların yüklenmesi için bekle
    except:
        print("🚫 Daha fazla yorum yok ya da buton bulunamadı.")
        break

# Yorumların yüklenmesini bekle
WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.TAG_NAME, "article"))
)

# Yorumları çek
articles = driver.find_elements(By.TAG_NAME, "article")
reviews = []
for article in articles:
    try:
        title = article.find_element(By.TAG_NAME, "h3").text.strip()
    except:
        title = ""
    try:
        content = article.find_element(By.CLASS_NAME, "content").text.strip()
    except:
        content = ""
    full_review = f"{title} {content}".strip()
    if full_review:
        reviews.append(full_review)

driver.quit()

# CSV'ye kaydet
df = pd.DataFrame(reviews, columns=["review"])
df.to_csv("imdb_shawshank_reviews.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ Toplam {len(df)} yorum başarıyla 'imdb_shawshank_reviews.csv' dosyasına kaydedildi.")
