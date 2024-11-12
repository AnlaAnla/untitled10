from docling.document_converter import DocumentConverter

source = r"C:\Users\martin\Downloads\Prompt.And.Hotwords.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"
