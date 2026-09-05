import urllib.request
import re

html = urllib.request.urlopen('https://documind-rag-git-main-kk036.vercel.app').read().decode()
js_files = re.findall(r'src="([^"]+\.js)"', html)

for js in js_files:
    url = js if js.startswith('http') else 'https://documind-rag-git-main-kk036.vercel.app' + (js if js.startswith('/') else '/' + js)
    js_content = urllib.request.urlopen(url).read().decode()
    if 'documind-api-4ww0' in js_content:
        print('API IS IN JS FILE:', js)
    elif 'localhost' in js_content:
        print('LOCALHOST IS IN JS FILE:', js)
