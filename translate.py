from googletrans import Translator

class TranslatorModule:
    def __init__(self):
        self.translator = Translator()

    def translate_text(self, text, src_lang, dest_lang):
        try:
            translation = self.translator.translate(text, src=src_lang, dest=dest_lang)
            return translation.text
        except Exception as e:
            print("An error occurred during translation:", e)
            return None