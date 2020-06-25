import os
import base64
import json
import requests
import shutil

from search import search_image

dataset_dir = 'new_data'

imgbb_key = "81da4f6a5de7e5db6b78c835a9f32167"

image_to_search = "recoverd.png"

ref_image_name = "ref" # don't add .png or .jpg etc.

for img_dir in os.listdir(dataset_dir):
	print("Getting reference image for img {}".format(img_dir))
	#### Upload img to imgbb.com
	with open(os.path.join(dataset_dir,img_dir,image_to_search), "rb") as file:
		api_url = "https://api.imgbb.com/1/upload"
		payload = {
			"expiration": 60,
			"key": imgbb_key,
			"image": base64.b64encode(file.read()),
		}
		res = requests.post(api_url, payload)

		json_data = json.loads(res.text)
		web_url = json_data['data']['url']

	#### Search and download reference image
	res = search_image(web_url, os.path.join(dataset_dir,img_dir), ref_image_name)
	if res == -1:
		print("FAILED!")
		shutil.rmtree(os.path.join(dataset_dir,img_dir))
