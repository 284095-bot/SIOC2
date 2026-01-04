import cv2
import numpy as np

def apply_convolution_filters(image_path):
    """
    Realizacja wymagania 1: Zaimplementować 3 podane zastosowania konwolucji (3.0).
    Zastosowania: Wykrywanie krawędzi, Rozmywanie, Wyostrzanie.
    """
    # Wczytanie obrazu w skali szarości
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Błąd: Nie można wczytać obrazu {image_path}")
        return

    # --- 1. Wykrywanie krawędzi (Operator Laplace'a) ---
    # Zgodnie z instrukcją używamy operatora L [cite: 8, 10]
    kernel_laplace = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ])
    
    # --- 2. Rozmywanie (Rozmycie Gaussowskie) ---
    # Zgodnie z instrukcją używamy jądra G z wagą 1/16 [cite: 16, 18]
    kernel_gauss = np.array([
        [1, 2, 1],
        [1, 4, 1],
        [1, 2, 1]
    ]) / 16.0
    
    # --- 3. Wyostrzanie ---
    # Zgodnie z instrukcją używamy jądra W [cite: 22, 24]
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    # Aplikacja filtrów za pomocą cv2.filter2D (konwolucja dyskretna) [cite: 5]
    edges = cv2.filter2D(img, -1, kernel_laplace)
    blurred = cv2.filter2D(img, -1, kernel_gauss)
    sharpened = cv2.filter2D(img, -1, kernel_sharpen)

    # Wyświetlenie wyników
    cv2.imshow("Oryginal", img)
    cv2.imshow("Krawedzie (Laplace)", edges)
    cv2.imshow("Rozmycie (Gauss)", blurred)
    cv2.imshow("Wyostrzanie", sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def demosaicing_bayer(bayer_img_path):
    """
    Realizacja wymagania 2: Wykonać demozaikowanie obrazów korzystając z konwolucji 2D dla filtru Bayera (3.0).
    Wzór Bayera przyjęty z macierzy B w instrukcji (GRBG):
    G R
    B G
    
    """
    # Wczytanie obrazu mozaiki (surowy obraz 1-kanałowy)
    raw = cv2.imread(bayer_img_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        print(f"Błąd: Nie można wczytać obrazu {bayer_img_path}")
        return

    raw = raw.astype(np.float32) / 255.0
    h, w = raw.shape

    # --- Tworzenie masek dla poszczególnych kolorów (GRBG) ---
    # Maska R (czerwony) - obecny w wierszu 0, kolumnie 1 (co 2)
    mask_r = np.zeros((h, w), dtype=np.float32)
    mask_r[0::2, 1::2] = 1

    # Maska B (niebieski) - obecny w wierszu 1, kolumnie 0 (co 2)
    mask_b = np.zeros((h, w), dtype=np.float32)
    mask_b[1::2, 0::2] = 1

    # Maska G (zielony) - obecny tam, gdzie nie ma R i B
    mask_g = np.zeros((h, w), dtype=np.float32)
    mask_g[0::2, 0::2] = 1 # G w rzędach parzystych
    mask_g[1::2, 1::2] = 1 # G w rzędach nieparzystych

    # Rozdzielenie kanałów (tam gdzie nie ma koloru, jest 0)
    R = raw * mask_r
    G = raw * mask_g
    B = raw * mask_b

    # --- Definicja jąder konwolucji do interpolacji ---
    # Zgodnie z instrukcją: wzmocnienie (gain) powinno być proporcjonalne do ilości pikseli.
    # Dla maski 2x2:
    # - Czerwony i Niebieski (1 piksel na 4): wymagane wzmocnienie 4.
    # - Zielony (2 piksele na 4): wymagane wzmocnienie 2.
    
    # Jądro dla R i B (wzmocnienie 4)
    # Macierz podstawowa sumuje się do 16. Dzielimy przez 4 -> suma 4.
    kernel_rb = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 4.0

    # Jądro dla G (wzmocnienie 2)
    # Macierz podstawowa sumuje się do 8. Dzielimy przez 4 -> suma 2.
    kernel_g = np.array([
        [0, 1, 0],
        [1, 4, 1],
        [0, 1, 0]
    ]) / 4.0

    # --- Wykonanie konwolucji (interpolacja brakujących pikseli) [cite: 48, 49] ---
    # Należy stosować padding zerami, aby uniknąć błędów na brzegach [cite: 54, 55]
    # W OpenCV BORDER_CONSTANT = 0 domyślnie przy odpowiedniej konfiguracji, ale warto wymusić.
    
    R_full = cv2.filter2D(R, -1, kernel_rb, borderType=cv2.BORDER_CONSTANT)
    G_full = cv2.filter2D(G, -1, kernel_g,  borderType=cv2.BORDER_CONSTANT)
    B_full = cv2.filter2D(B, -1, kernel_rb, borderType=cv2.BORDER_CONSTANT)

    # Złożenie obrazu RGB
    merged = cv2.merge([B_full, G_full, R_full]) # OpenCV używa kolejności BGR
    merged = np.clip(merged, 0, 1) # Zabezpieczenie zakresu [cite: 52]

    # Wyświetlenie
    cv2.imshow("Obraz zdemozaikowany", merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Uruchomienie ---
if __name__ == "__main__":
    # Podmień nazwy plików na te, które masz w pliku ZIP
    print("Uruchamianie czesci 1: Podstawowe filtry")
    # apply_convolution_filters("nazwa_obrazka.jpg") 
    
    print("Uruchamianie czesci 2: Demozaikowanie")
    # demosaicing_bayer("nazwa_mozaiki.bmp")