import cv2
import numpy as np
import urllib.request

def main_pipeline():
    url = "https://github.com/284095-bot/SIOC2/raw/main/Seal.jpg"
    
    try:
        resp = urllib.request.urlopen(url)
        img_array = np.array(bytearray(resp.read()), dtype=np.uint8)
        img_color_src = cv2.imdecode(img_array, -1)
    except:
        return

    if img_color_src is None:
        return

    max_height = 350
    h, w = img_color_src.shape[:2]
    scale = max_height / h
    new_w = int(w * scale)
    
    new_w = new_w - (new_w % 6)
    max_height = max_height - (max_height % 6)
    
    img_color_src = cv2.resize(img_color_src, (new_w, max_height))
    img_gray = cv2.cvtColor(img_color_src, cv2.COLOR_BGR2GRAY)

    k_sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    k_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = cv2.filter2D(img_gray.astype(np.float64), -1, k_sobel_x)
    sobel_y = cv2.filter2D(img_gray.astype(np.float64), -1, k_sobel_y)
    res_sobel = np.clip(np.abs(sobel_x) + np.abs(sobel_y), 0, 255).astype(np.uint8)

    k_laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    res_laplace = cv2.filter2D(img_gray, -1, k_laplace)

    k_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    k_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewitt_x = cv2.filter2D(img_gray.astype(np.float64), -1, k_prewitt_x)
    prewitt_y = cv2.filter2D(img_gray.astype(np.float64), -1, k_prewitt_y)
    res_prewitt = np.clip(np.abs(prewitt_x) + np.abs(prewitt_y), 0, 255).astype(np.uint8)

    k_scharr_x = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
    k_scharr_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])
    scharr_x = cv2.filter2D(img_gray.astype(np.float64), -1, k_scharr_x)
    scharr_y = cv2.filter2D(img_gray.astype(np.float64), -1, k_scharr_y)
    res_scharr = np.clip(np.abs(scharr_x) + np.abs(scharr_y), 0, 255).astype(np.uint8)

    k_gauss = np.array([[1, 2, 1], [1, 4, 1], [1, 2, 1]]) / 16.0
    res_gauss = cv2.filter2D(img_color_src, -1, k_gauss)

    k_box = np.ones((3, 3), dtype=np.float32) / 9.0
    res_box = cv2.filter2D(img_color_src, -1, k_box)

    k_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    res_sharpen = cv2.filter2D(img_color_src, -1, k_sharpen)

    mask_b_bay = np.array([[0, 0], [1, 0]], dtype=np.uint8)
    mask_g_bay = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    mask_r_bay = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    bayer_mask_small = cv2.merge([mask_b_bay, mask_g_bay, mask_r_bay])

    bayer_filter = np.tile(bayer_mask_small, (img_color_src.shape[0] // 2, img_color_src.shape[1] // 2, 1))
    sensor_image_bayer = img_color_src.astype(np.float32) * bayer_filter

    k_gain_4 = np.ones((2, 2), dtype=np.float32)
    k_gain_2 = np.ones((2, 2), dtype=np.float32) * 0.5
    
    rec_channels_bayer = []
    for c in range(3):
        layer = sensor_image_bayer[:, :, c]
        k = k_gain_2 if c == 1 else k_gain_4
        rec_channels_bayer.append(cv2.filter2D(layer, -1, k, borderType=cv2.BORDER_CONSTANT))

    res_bayer = np.clip(cv2.merge(rec_channels_bayer), 0, 255).astype(np.uint8)

    fuji_g = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.uint8)
    fuji_b = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.uint8)
    fuji_r = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)

    mask_ch0_fuji = fuji_b
    mask_ch1_fuji = fuji_g
    mask_ch2_fuji = fuji_r

    fuji_mask_small = cv2.merge([mask_ch0_fuji, mask_ch1_fuji, mask_ch2_fuji])
    
    tiles_y_f = img_color_src.shape[0] // 3
    tiles_x_f = img_color_src.shape[1] // 3
    fuji_filter = np.tile(fuji_mask_small, (tiles_y_f, tiles_x_f, 1))
    
    sensor_image_fuji = img_color_src.astype(np.float32) * fuji_filter

    k_fuji_g = np.ones((3, 3), dtype=np.float32) * (9.0 / 5.0)
    k_fuji_rb = np.ones((3, 3), dtype=np.float32) * (9.0 / 2.0)

    rec_channels_fuji = []
    for c in range(3):
        layer = sensor_image_fuji[:, :, c]
        k = k_fuji_g if c == 1 else k_fuji_rb
        k_norm = k / 9.0 
        rec_channels_fuji.append(cv2.filter2D(layer, -1, k_norm, borderType=cv2.BORDER_CONSTANT))

    res_fuji = np.clip(cv2.merge(rec_channels_fuji), 0, 255).astype(np.uint8)

    def to_vis(gray_img):
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    def label(img, text):
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    v_org = img_color_src
    v_sobel = to_vis(res_sobel); label(v_sobel, "Sobel (X+Y)")
    v_laplace = to_vis(res_laplace); label(v_laplace, "Laplace")
    v_prewitt = to_vis(res_prewitt); label(v_prewitt, "Bonus: Prewitt")
    v_scharr = to_vis(res_scharr); label(v_scharr, "Bonus: Scharr")
    
    v_gauss = res_gauss; label(v_gauss, "Gauss Blur")
    v_box = res_box; label(v_box, "Box Blur")
    v_sharp = res_sharpen; label(v_sharp, "Wyostrzanie")

    v_bay_sens = sensor_image_bayer.astype(np.uint8); label(v_bay_sens, "Bayer RAW")
    v_bay_res = res_bayer; label(v_bay_res, "Bayer Wynik")
    
    v_fuji_sens = sensor_image_fuji.astype(np.uint8); label(v_fuji_sens, "Bonus: Fuji RAW")
    v_fuji_res = res_fuji; label(v_fuji_res, "Bonus: Fuji Wynik")

    win1 = np.vstack([np.hstack([v_org, v_sobel]), np.hstack([v_laplace, v_prewitt]), np.hstack([v_scharr, np.zeros_like(v_org)])])
    win2 = np.hstack([v_gauss, v_box, v_sharp])
    win3 = np.hstack([v_org, v_bay_sens, v_bay_res])
    win4 = np.hstack([v_org, v_fuji_sens, v_fuji_res])

    cv2.imshow("1. Zadanie: Krawedzie (PDF s.1)", win1)
    cv2.imshow("2. Zadanie: Rozmycie i Wyostrzanie (PDF s.1-2)", win2)
    cv2.imshow("3. Zadanie: Demozaikowanie Bayera (PDF s.2)", win3)
    cv2.imshow("4. Zadanie: Demozaikowanie Fuji (Bonus)", win4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_pipeline()
