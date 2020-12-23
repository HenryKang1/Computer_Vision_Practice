from selenium import webdriver
from selenium.webdriver.common.keys import Keys
#from selenium.webdriver import option
from selenium.webdriver.chrome.options import Options

import os
import time

import urllib.request

def Run():
    chrome_options = Options()
    chrome_options.add_experimental_option("detach", True)
    #chrome_options.binary_location="D:\web_coding_study\selenium\chromedriver.exe"
    driverLocation="D:\web_coding_study\selenium\\chromedriver.exe"
    #os.environ["webdriver.chrome.driver"] = driverLocation
    chrome_options.add_argument("disable-infobars");
    driver = webdriver.Chrome(executable_path=driverLocation,chrome_options=chrome_options)
    driver.get("https://www.google.ca/imghp?hl=en&tab=ri&authuser=0&ogbl")
    elem= driver.find_element_by_name("q")
    #time.sleep(3)
    elem.send_keys("healty meal")
    elem.send_keys(Keys.RETURN)
    time.sleep(3)
    SCROLL_PAUSE_TIME = 2

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                dirver.find_element_by_css_selector(".mye4qd").click()
            except:
                break

        last_height = new_height
    images=driver.find_elements_by_css_selector(".rg_i.Q4LuWd")#[0].click()
    time.sleep(10)
    count=1
    for image in images:
        try:
            image.click()
            time.sleep(3)
            #imgUrl=driver.find_element_by_css_selector(".n3VNCb").get_attribute("src")
            imgUrl=driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img').get_attribute("src")
            
            urllib.request.urlretrieve(imgUrl,"%s.jpg"%count)
            count=count+1
        except:
            pass

    #while(True):
    #    pass

import os
path=os.getcwd()
img_path=path+"/image/"
if img_path is None:

    os.mkdir(image)

Run()