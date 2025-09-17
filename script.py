import os
import webbrowser
import mysql.connector
from datetime import date
import ssl
import serial
import time

arduino_port = "/dev/ttyUSB0"
serial_port = '/dev/ttyUSB0'
baud_rate = 9600
context = ssl.create_default_context()
# MySQL veritabanı bağlantı bilgileri
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWD = "1234"
DB_DATABASE = "hpmyadmin"

def clear_screen():
    """Ekranı temizler."""
    os.system('cls' if os.name == 'nt' else 'clear')

def connect_db():
    """Veritabanına bağlanır ve bağlantı nesnesini döndürür."""
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            passwd=DB_PASSWD,
            database=DB_DATABASE
        )
        return connection
    except mysql.connector.Error as err:
        print(f"Veritabanına bağlanırken bir hata oluştu: {err}")
        return None


def login():
    """Kullanıcı girişi yapar ve başarılı olursa True, aksi halde False döner."""
    clear_screen()
    print("----- KULLANICI GİRİŞİ -----")
    username = input("Kullanıcı Adınızı Giriniz: ")
    userpass = input("Şifrenizi Giriniz: ")

    connection = connect_db()
    if connection is None:
        return False

    try:
        cursor = connection.cursor()

        # Adım 1: Sadece kullanıcı adının varlığını kontrol et
        query_username = "SELECT username FROM users WHERE username = %s"
        cursor.execute(query_username, (username,))
        username_exists = cursor.fetchone()

        if not username_exists:
            print("\nKullanıcı adı bulunamadı!")
            return False

        # Adım 2: Eğer kullanıcı adı varsa, şifrenin doğru olup olmadığını kontrol et
        query_password = "SELECT username FROM users WHERE username = %s AND password = %s"
        cursor.execute(query_password, (username, userpass))
        result = cursor.fetchone()

        if result:
            print(f"\nGiriş başarılı! Hoş geldiniz, {result[0]}.")
            time.sleep(3)
            userpage()
        else:
            print("\nŞifre hatalı!")
            return False

    except mysql.connector.Error as err:
        print(f"Sorgu çalıştırılırken bir hata oluştu: {err}")
        return False
    finally:
        # Bağlantıyı her zaman kapat
        if connection and connection.is_connected():

            cursor.close()
            connection.close()



# Diğer fonksiyonlarınız ve ana menü kodunuz değişmeden kalabilir.

today = date.today()
print(today.strftime("%Y-%m-%d"))

def blink():
    ser = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(0.5)
    ser.write(b'L')
    ser.close()

def buzz():
    ser = serial.Serial(arduino_port, 9600, timeout=1)
    time.sleep(1)
    ser.write(b'B')
    ser.close()


def lightcontrol():
    os.system("clear")
    print("BU BÖLÜM DAHA TAMAMLANMAMIŞTIR")
    time.sleep(3)
    print("Geri Dönülüyor...")
    time.sleep(1)

def get_temperature_from_arduino(port='/dev/ttyUSB0', baud_rate=9600):
    line = None  # Başlangıçta boş bir değer tanımla
    try:
        # Seri port bağlantısını kur
        ser = serial.Serial(port, baud_rate, timeout=2)
        time.sleep(2)  # Bağlantının kurulması için bekle

        # Gelen ilk satırı oku
        line = ser.readline().decode('utf-8').strip()

    except serial.SerialException as e:
        print(f"Seri port hatası: {e}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
    finally:
        # Seri portu kapat
        if 'ser' in locals() and ser.is_open:
            ser.close()
            # print("Seri port kapatıldı.")  # İsteğe bağlı, debug için kaldırılabilir

    return line  # Okunan değeri döndür

def fancontrol():
    os.system("clear")
    get_temperature_from_arduino(port='/dev/ttyUSB0')
    sicaklik = get_temperature_from_arduino(port='/dev/ttyUSB0')

    print("Şu anki Sıcaklık:", sicaklik, "Derece")

    while True:
        try:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)
            fancontrolinput = int(input("Fan açmak için 1 Fan kapatmak için 2 yazınız Geri Dönmek için 3 Yazınız : "))

            if fancontrolinput == 1:
                ser.write(b'F')
                print("FAN AÇILDI")
                blink()
                buzz()
                time.sleep(1)
            elif fancontrolinput == 2:
                ser.write(b'G')
                print("FAN KAPATILDI")
                blink()
                buzz()
                time.sleep(1)
            elif fancontrolinput == 3:
                break;

        except ValueError:
            # Sayısal olmayan girişler için uyarı
            print("\nGeçersiz giriş! Lütfen bir sayı girin.")


def tempcontrol():

    os.system("clear")
    countdown = 4

    get_temperature_from_arduino(port='/dev/ttyUSB0')

    # Başlangıç sıcaklık değerlerini tanımlama
    sicaklik = get_temperature_from_arduino(port='/dev/ttyUSB0')  # Sensörden gelen mevcut sıcaklık (örnek)
    ayarli_sicaklik = 32  # Ayarlanan sıcaklık için başlangıç değeri



    print("Şu anki Sıcaklık:", sicaklik, "Derece")
    print("Ayarlanan Sıcaklık:", ayarli_sicaklik, "Derece")

    while True:
        try:
            ser = serial.Serial(arduino_port, 9600, timeout=1)
            time.sleep(2)
            istenilen_sicaklik = int(input("İstenilen Sıcaklık Değerini Giriniz (MAX 40 - MIN 20): "))

            if 20 <= istenilen_sicaklik <= 40:
                ayarli_sicaklik = istenilen_sicaklik
                print("\nSıcaklık Değeri", ayarli_sicaklik, "olarak ayarlandı!")
                blink()
                # Güncelleme sonrası çıktıyı ekrana yazdırıp döngüyü sonlandır.
                print("Yeni Ayarlanan Sıcaklık:", ayarli_sicaklik, "Derece")
                buzz()
                time.sleep(2)
                os.system("cls" if os.name == "nt" else "clear")
                ser.close()
                break  # Döngüden çık
            else:
                # Geçersiz değer için uyarı
                ser = serial.Serial(arduino_port, 9600, timeout=1)
                print("\nYanlış bir değer girdiniz! Lütfen 20 ile 40 arasında bir sayı girin.")
                for countdown in range(4,0,1):
                    ser.write(b'B')
                    time.sleep(0.1)
                    ser.write(b'S')
                    time.sleep(0.1)

        except ValueError:
            # Sayısal olmayan girişler için uyarı
            print("\nGeçersiz giriş! Lütfen bir sayı girin.")


def userpage():
    while True:
        os.system("clear")
        print("*---ÇELTİK KURUTMA OTOMASYONU---* \n             SÜRÜM 0.1")
        print("")
        print("")
        print("")
        havasicakligi = 27
        os.system("figlet HOS GELDINIZ")
        print("SICAKLIK : " , havasicakligi)
        print("Tarih : " , (today.strftime("%Y-%m-%d")))
        print("1- Sıcaklık Ayarı")
        print("2- Işık Kontrol")
        print("3- Fan Kontrol")
        print("4- Çıkış")
        islemtercih = input("Isleminizi Seciniz :")
        if islemtercih == "1":
            tempcontrol()
            islemtercih == 0
        elif islemtercih == "2":
            lightcontrol()
        elif islemtercih == "3":
            fancontrol()
        elif islemtercih == "4":
            os.system("clear")
            login()
            break

def main_menu():
    """Ana menüyü gösterir ve kullanıcıdan seçim alır."""
    while True:
        clear_screen()
        print("----- ÇELTİK KURUTMA OTOMASYONU -----")
        print("1- Kullanıcı Girişi")
        print("2- Youtube")
        print("3- Program Hakkında")
        print("4- Çıkış")
        print("-" * 20)

        choice = input("Hangi işlemi yapmak istersiniz? ")
        print()
        if choice == "1":
            login()
        elif choice == "2":
            print("Youtube Açılıyor...")
            time.sleep(2)
            webbrowser.open('https://www.youtube.com')
        elif choice == "3":
            os.system("clear")
            os.system("figlet HAKKIMIZDA")
            time.sleep(1)
            print("Çeltik Kurutma Makine Otomasyonu\n")
            time.sleep(1)
            print("Yağız ŞEN")
            time.sleep(1)
            print("\n 2025 Tüm Hakları Saklıdır")
            time.sleep(30)
        elif choice == "4":
            print("Programdan çıkılıyor...")
            break
        else:
            print("Geçersiz seçim. Lütfen tekrar deneyin.")
            time.sleep(2)

# Programı başlatmak için ana menüyü çağır
#main_menu()
login()
