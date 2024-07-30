from cnocr import CnOcr

img_fp = r"C:\Code\ML\Image\Card_test\test03\18fa2d4595c9b6a998f3f2be793a6f9.jpg"
ocr = CnOcr(det_model_name='en_PP-OCRv3_det', rec_model_name='ch_PP-OCRv3')
out = ocr.ocr(img_fp)

print(out)
print()