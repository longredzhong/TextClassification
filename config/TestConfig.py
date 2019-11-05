from config.BaseConfig import BaseConfig
class TestConfig(BaseConfig):
    CharVectors = None
    WordVectors = None
    CharVectorsDim = 300
    WordVectorsDim = 300
    CharVocabSize = None
    WordVocabSize = None

if __name__ == "__main__":
    a = TestConfig()
    