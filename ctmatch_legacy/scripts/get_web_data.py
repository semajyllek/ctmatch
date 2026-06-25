
from selenium import webdriver

def save_web_data(url: str) -> None:
	driver = webdriver.Chrome()
	driver.get(url)
	button = driver.find_element_by_class_name("save-list")
	button.click()


if __name__ == "__main__":
	url = "https://clinicaltrials.gov/ct2/results?cond=Heart+Diseases"
	save_web_data(url)

