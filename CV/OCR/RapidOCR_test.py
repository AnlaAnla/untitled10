from rapidocr import RapidOCR

engine = RapidOCR()

img_url = r"C:\Code\ML\Project\CardVideoSummary\static\frames\77a4b046-e107-4468-9f1a-5b0e9a3446f8_5813627.jpg"
result = engine(img_url)
print(result)

result.vis("vis_result.jpg")

