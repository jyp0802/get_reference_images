# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 16:48:55 2020

@author: User
"""

#!/usr/bin/env python3

# requires: selenium, chromium-driver, retry
# requires: selenium, chromium-driver, retry

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import selenium.common.exceptions as sel_ex
import os
import sys
import time
import urllib.parse
from retry import retry
import argparse
import logging
from urllib.request import urlretrieve

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger()
retry_logger = None


css_thumbnail = "img.M4dUYb"
#css_thumbnail = "img.rISBZc"
#css_thumbnail = "img.Q4LuWd"
css_large = "img.n3VNCb"
css_load_more = ".mye4qd"
selenium_exceptions = (sel_ex.ElementClickInterceptedException, sel_ex.ElementNotInteractableException, sel_ex.StaleElementReferenceException)

def scroll_to_end(wd):
    wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")

@retry(exceptions=KeyError, tries=6, delay=0.1, backoff=2, logger=retry_logger)
def get_thumbnails(wd, want_more_than=0):
    thumbnails = wd.find_elements_by_css_selector(css_thumbnail)
    n_results = len(thumbnails)
    if n_results <= want_more_than:
        raise KeyError("no new thumbnails")
    return thumbnails

@retry(exceptions=KeyError, tries=6, delay=0.1, backoff=2, logger=retry_logger)
def get_image_src(wd):
    actual_images = wd.find_elements_by_css_selector(css_large)
    sources = []
    for img in actual_images:
        src = img.get_attribute("src")
        if src.startswith("http") and not src.startswith("https://encrypted-tbn0.gstatic.com/"):
            sources.append(src)
    if not len(sources):
        raise KeyError("no large image")
    return sources

@retry(exceptions=selenium_exceptions, tries=6, delay=0.1, backoff=2, logger=retry_logger)
def retry_click(el):
    el.click()

def get_images(wd, query, save_dir, save_name, start=0, n=1):
    thumbnails = []
    count = len(thumbnails)
    while count < n:
        try:
            thumbnails = get_thumbnails(wd, want_more_than=count)
        except KeyError as e:
            logger.warning("cannot load enough thumbnails")
            break
        count = len(thumbnails)
    sources = []
    retry_num = 0
    for tn in thumbnails:
        if retry_num > 5:
            return -1
        try:
            retry_click(tn)
        except selenium_exceptions as e:
            logger.warning("main image click failed")
            retry_num += 1
            continue
        sources1 = []
        try:
            sources1 = get_image_src(wd)
            if not sources1:
                tn_src = tn.get_attribute("src")
                if not tn_src.startswith("data"):
                    logger.warning("no src found for main image, using thumbnail")          
                    sources1 = [tn_src]
                else:
                    logger.warning("no src found for main image, thumbnail is a data URL")
            for src in sources1:
                if not src in sources:
                    sources.append(src)
                    urlretrieve(src, "{}/{}".format(save_dir,save_name))
        except KeyError as e:
            pass
        
        if len(sources) >= n:
            break
    if len(sources) == 0:
        return -1
    return sources

def google_image_search(wd, query, save_dir, save_name, safe="off", n=20, opts=''):
    search_url_t='https://www.google.com/searchbyimage?hl=en-US&image_url={q}'
    search_url = search_url_t.format(q=urllib.parse.quote(query))
    print(search_url)
    wd.get(search_url)
    sources = get_images(wd, query, save_dir, save_name, n=n)
    return sources

def search_image(img_url, save_dir, save_name):
    opts = Options()
    opts.add_argument("--headless")
    with webdriver.Chrome(options=opts, executable_path='/home/junyong/chromedriver') as wd:
        sources = google_image_search(wd, img_url, save_dir, save_name, safe="off", n=1, opts="")
    return sources
