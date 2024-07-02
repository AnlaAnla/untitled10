import easyocr

reader = easyocr.Reader(['en', 'ch_sim'])
result = reader.readtext(r"C:\Code\ML\Image\Card_test\test03\25792694ce3604eb89eed852b08efa9.jpg")
print(result)