import urllib.request, json

url = "http://dentaltw-info.uc.r.appspot.com/labels/completed"

response = urllib.request.urlopen(url)

data = json.loads(response.read())

print (data)