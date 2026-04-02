from rapidocr import RapidOCR

engine = RapidOCR()

img_url = r"C:\Code\ML\Project\CardVideoSummary\static\frames\7eb64157-3ad6-4014-a9cb-61e7b708f790_5754600.jpg"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")

