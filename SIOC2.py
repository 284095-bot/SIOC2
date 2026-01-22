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

    mask_r = np.zeros((max_height, new_w), dtype=np.float32)
    mask_g = np.zeros((max_height, new_w), dtype=np.float32)
    mask_b = np.zeros((max_height, new_w), dtype=np.float32)

    mask_g[0::2, 0::2] = 1
    mask_r[0::2, 1::2] = 1
    mask_b[1::2, 0::2] = 1
    mask_g[1::2, 1::2] = 1

    s_r = img_color_src[:, :, 2] * mask_r
    s_g = img_color_src[:, :, 1] * mask_g
    s_b = img_color_src[:, :, 0] * mask_b

    k_b_rb = np.ones((2, 2), dtype=np.float32)
    k_b_g = np.ones((2, 2), dtype=np.float32) * 0.5

    rb_bayer = cv2.filter2D(s_r, -1, k_b_rb, borderType=cv2.BORDER_CONSTANT)
    gb_bayer = cv2.filter2D(s_g, -1, k_b_g, borderType=cv2.BORDER_CONSTANT)
    bb_bayer = cv2.filter2D(s_b, -1, k_b_rb, borderType=cv2.BORDER_CONSTANT)

    res_bayer = cv2.merge([
        np.clip(bb_bayer, 0, 255).astype(np.uint8),
        np.clip(gb_bayer, 0, 255).astype(np.uint8),
        np.clip(rb_bayer, 0, 255).astype(np.uint8)
    ])

    v_bay_sens = cv2.merge([(s_b).astype(np.uint8), (s_g).astype(np.uint8), (s_r).astype(np.uint8)])

    f_g_p = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
    f_b_p = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=np.float32)
    f_r_p = np.array([[0, 0, 1], [1, 0, 0], [0, 0, 0]], dtype=np.float32)

    mask_f_g = np.tile(f_g_p, (max_height // 3, new_w // 3))
    mask_f_b = np.tile(f_b_p, (max_height // 3, new_w // 3))
    mask_f_r = np.tile(f_r_p, (max_height // 3, new_w // 3))

    s_f_r = img_color_src[:, :, 2] * mask_f_r
    s_f_g = img_color_src[:, :, 1] * mask_f_g
    s_f_b = img_color_src[:, :, 0] * mask_f_b

    k_f_rb = np.ones((3, 3), dtype=np.float32) * (9.0 / 2.0)
    k_f_g = np.ones((3, 3), dtype=np.float32) * (9.0 / 5.0)

    rf_fuji = cv2.filter2D(s_f_r, -1, k_f_rb / 9.0, borderType=cv2.BORDER_CONSTANT) * 9.0
    gf_fuji = cv2.filter2D(s_f_g, -1, k_f_g / 9.0, borderType=cv2.BORDER_CONSTANT) * 9.0
    bf_fuji = cv2.filter2D(s_f_b, -1, k_f_rb / 9.0, borderType=cv2.BORDER_CONSTANT) * 9.0

    res_fuji = cv2.merge([
        np.clip(bf_fuji, 0, 255).astype(np.uint8),
        np.clip(gf_fuji, 0, 255).astype(np.uint8),
        np.clip(rf_fuji, 0, 255).astype(np.uint8)
    ])

    v_fuji_sens = cv2.merge([(s_f_b).astype(np.uint8), (s_f_g).astype(np.uint8), (s_f_r).astype(np.uint8)])

    def to_vis(gray_img):
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    def label(img, text):
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    v_org = img_color_src
    v_sobel = to_vis(res_sobel); label(v_sobel, "Sobel")
    v_laplace = to_vis(res_laplace); label(v_laplace, "Laplace")
    v_prewitt = to_vis(res_prewitt); label(v_prewitt, "Prewitt")
    v_scharr = to_vis(res_scharr); label(v_scharr, "Scharr")
    
    v_gauss = res_gauss; label(v_gauss, "Gauss")
    v_box = res_box; label(v_box, "Jadro usredniajace")
    v_sharp = res_sharpen; label(v_sharp, "Wyostrzanie")

    label(v_bay_sens, "Bayer Mozaika")
    v_bay_res = res_bayer; label(v_bay_res, "Bayer Demozaikowanie")
    
    label(v_fuji_sens, "Fuji Mozaika")
    v_fuji_res = res_fuji; label(v_fuji_res, "Fuji Demozaikowanie")

    win1 = np.vstack([np.hstack([v_org, v_sobel]), np.hstack([v_laplace, v_prewitt]), np.hstack([v_scharr, np.zeros_like(v_org)])])
    win2 = np.hstack([v_gauss, v_box, v_sharp])
    win3 = np.hstack([v_org, v_bay_sens, v_bay_res])
    win4 = np.hstack([v_org, v_fuji_sens, v_fuji_res])

    cv2.imshow("1. Zadanie: Wykrywanie krawedzi", win1)
    cv2.imshow("2. Zadanie: Rozmycie i Wyostrzanie", win2)
    cv2.imshow("3. Zadanie: Demozaikowanie Bayera", win3)
    cv2.imshow("4. Zadanie: Demozaikowanie Fuji", win4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_pipeline()
