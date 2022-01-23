import urllib.request, json

url = "http://dentaltw-info.uc.r.appspot.com/labels/completed"
response = urllib.request.urlopen(url)
tm_text = json.loads(response.read())
with open('data.json','w') as f:
    json.dump(tm_text, f)