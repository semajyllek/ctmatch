
import pytest

from gen_categories import gen_category, CAT_GEN_MODEL
from transformers import pipeline

s1 = "A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain that began two days earlier for the first time in her life. The pain started while she was walking, radiates to the back, and is accompanied by nausea, diaphoresis and mild dyspnea, but is not increased on inspiration. The latest episode of pain ended half an hour prior to her arrival. She is known to have hypertension and obesity. She denies smoking, diabetes, hypercholesterolemia, or a family history of heart disease. She currently takes no medications. Physical examination is normal. The EKG shows nonspecific changes."
s2 = "64-year-old obese female with diagnosis of diabetes mellitus and persistently elevated HbA1c. She is reluctant to see a nutritionist and is not compliant with her diabetes medication or exercise. She complains of a painful skin lesion on the left lower leg. She has tried using topical lotions and creams but the lesion has increased in size and is now oozing."
test_pairs = [
    (s1, 'cardiac'), 
    (s2, 'endocrine')
]

# instantiate llm pipeline
pipe = pipeline(CAT_GEN_MODEL)

@pytest.mark.parametrize("test_input,expected", test_pairs)
def test_gen_cat(test_input, expected):
    assert gen_category(pipe, test_input) == expected
