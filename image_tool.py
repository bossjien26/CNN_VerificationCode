from PIL import Image
import os

# im = Image.open('./train_images/aaii_22.png')

# print(im.size)
# print(type(im.size))
# # (400, 225)
# # <class 'tuple'>

# w, h = im.size
# print('width: ', w)
# print('height:', h)
# width:  400
# height: 225

image_name_list = os.listdir('./train_images')

for index, image_name in enumerate(image_name_list):
    im = Image.open('./train_images/'+image_name)
    name = image_name.split('_')[0]
    im.save('./template/'+name+'_'+str(index)+'.png')



# import requests
# from xml.etree import ElementTree
# import base64
# import cv2
# from matplotlib import pyplot as plt
# import os

# headers = {
#     'Pragma':'no-cache',
# 'Referrer-Policy':'no-referrer-when-downgrade',
# 'Strict-Transport-Security':'max-age=31536000; includeSubDomains',
# 'X-Content-Type-Options':'nosniff',
# 'X-Frame-Options':'DENY',
# 'X-Xss-Protection':'1;mode=block',
# 'Connection':'keep-alive',
# 'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
# 'Content-Security-Policy':"default-src 'self' 'unsafe-eval' https://jsonip.com/; style-src 'self' 'unsafe-inline'; media-src *; script-src 'self' 'unsafe-inline' 'unsafe-eval'; img-src 'self' *.blob.core.windows.net data: https:; font-src 'self' data:",
# 'Cookie':'ARRAffinity=0ec0fc7ca5c78d3a2efcafc11d44e490777473cff16796c588705d3a68cd3501; ARRAffinitySameSite=0ec0fc7ca5c78d3a2efcafc11d44e490777473cff16796c588705d3a68cd3501; UnobtrusiveSessionId=b12f5ac7-91ca-4ddd-90cc-1206065194ef'
# }
# path = './download_image'
# for i in range(5000):
#     fileName = str(i)+'.png'
#     with open(path+fileName, 'wb') as handle:
#         imagerequest = requests.get('https://aseconnect.aseglobal.com/api/user/GenerateValidationCode'
#             , headers=headers)
#         # tree = ElementTree.fromstring(imagerequest.content)
#         # print(imagerequest.text)
#         decoded_data=base64.b64decode((imagerequest.text))
#         #write the decoded data back to original format in  file
#         # img_file = open('image.jpeg', 'wb')
#         handle.write(decoded_data)
#         handle.close()
        
#         src_filepath = path+fileName
#         img = cv2.imread(src_filepath)
#         dst = cv2.fastNlMeansDenoisingColored(img,None,30,10,7,21)
#         cv2.imwrite(os.path.join(path, fileName), dst)
