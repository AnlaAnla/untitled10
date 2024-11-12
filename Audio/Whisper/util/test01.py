import ctranslate2


translator = ctranslate2.Translator(r"D:\Code\ML\Model\Lora\checkpoint-100-2024Y_10M_08D_17h_43m_12s\adapter_model")
translator.translate_batch(tokens)

generator = ctranslate2.Generator(generation_model_path)
generator.generate_batch(start_tokens)
