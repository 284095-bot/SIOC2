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

    max_height = 400
    h, w = img_color_src.shape[:2]
    scale = max_height / h
    new_w = int(w * scale)
    img_color_src = cv2.resize(img_color_src, (new_w, max_height))

    img_gray = cv2.cvtColor(img_color_src, cv2.COLOR_BGR2GRAY)

    kernel_laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_gauss = np.array([[1, 2, 1], [1, 4, 1], [1, 2, 1]]) / 16.0
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    res_edges = cv2.filter2D(img_gray, -1, kernel_laplace)
    res_blur = cv2.filter2D(img_gray, -1, kernel_gauss)
    res_sharpen = cv2.filter2D(img_gray, -1, kernel_sharpen)

    h, w = img_gray.shape
    raw_mosaic = np.zeros((h, w), dtype=np.float32)
    src_float = img_color_src.astype(np.float32) / 255.0

    raw_mosaic[0::2, 0::2] = src_float[0::2, 0::2, 1] 
    raw_mosaic[1::2, 1::2] = src_float[1::2, 1::2, 1] 
    raw_mosaic[0::2, 1::2] = src_float[0::2, 1::2, 2] 
    raw_mosaic[1::2, 0::2] = src_float[1::2, 0::2, 0] 

    mask_r = np.zeros((h, w), dtype=np.float32)
    mask_r[0::2, 1::2] = 1 
    
    mask_b = np.zeros((h, w), dtype=np.float32)
    mask_b[1::2, 0::2] = 1 
    
    mask_g = np.zeros((h, w), dtype=np.float32)
    mask_g[0::2, 0::2] = 1
    mask_g[1::2, 1::2] = 1 

    R_in = raw_mosaic * mask_r
    G_in = raw_mosaic * mask_g
    B_in = raw_mosaic * mask_b

    kernel_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0
    kernel_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]]) / 4.0

    R_out = cv2.filter2D(R_in, -1, kernel_rb, borderType=cv2.BORDER_CONSTANT)
    G_out = cv2.filter2D(G_in, -1, kernel_g,  borderType=cv2.BORDER_CONSTANT)
    B_out = cv2.filter2D(B_in, -1, kernel_rb, borderType=cv2.BORDER_CONSTANT)

    res_demosaic = cv2.merge([B_out, G_out, R_out])
    res_demosaic = np.clip(res_demosaic, 0, 1)
    res_demosaic_uint8 = (res_demosaic * 255).astype(np.uint8)

    def to_vis(gray_img):
        return cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    vis_gray_org = to_vis(img_gray)
    vis_edges = to_vis(res_edges)
    vis_blur = to_vis(res_blur)
    vis_sharpen = to_vis(res_sharpen)
    vis_mosaic_input = to_vis((raw_mosaic * 255).astype(np.uint8))

    def label(img, text):
        font_scale = 0.7 
        thickness = 2
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness + 2)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)

    label(vis_gray_org, "1. Obraz Oryginalny")
    label(vis_edges,    "2. Wykrywanie Krawedzi (Laplace)")
    label(vis_blur,     "3. Rozmywanie (Gauss)")
    label(vis_sharpen,  "4. Wyostrzanie")
    label(vis_mosaic_input, "5. Mozaika")
    label(res_demosaic_uint8, "6. Wynik Demozaikowania")

    row1 = np.hstack([vis_gray_org, vis_edges, vis_blur])
    row2 = np.hstack([vis_sharpen, vis_mosaic_input, res_demosaic_uint8])
    final_grid = np.vstack([row1, row2])

    window_name = "Zadanie - Pelny Ekran"
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cv2.imshow(window_name, final_grid)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_pipeline()
