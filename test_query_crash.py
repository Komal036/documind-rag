import urllib.request
import json
import urllib.error

# Login
data = json.dumps({'email':'kumarikomal3434@gmail.com','password':'password'}).encode()
req = urllib.request.Request('https://documind-api-4ww0.onrender.com/api/v1/auth/login', data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req)
token = json.loads(resp.read())['access_token']

# Upload big file
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
text = 'This is a very long text to ensure it gets chunked. ' * 500
body = (
    '--' + boundary + '\r\n'
    'Content-Disposition: form-data; name="file"; filename="test_big.txt"\r\n'
    'Content-Type: text/plain\r\n\r\n'
    + text + '\r\n'
    '--' + boundary + '--\r\n'
).encode('utf-8')

req = urllib.request.Request(
    'https://documind-api-4ww0.onrender.com/api/v1/ingest',
    data=body,
    headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Authorization': f'Bearer {token}'
    }
)
try:
    resp = urllib.request.urlopen(req)
    print('Ingest success:', resp.read().decode())
except urllib.error.HTTPError as e:
    print('Ingest error:', e.code, e.read().decode())

# Query
body = json.dumps({'question':'ensure it gets chunked','use_reranker':True}).encode()
req = urllib.request.Request('https://documind-api-4ww0.onrender.com/api/v1/query', data=body, headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {token}'})
try:
    resp = urllib.request.urlopen(req)
    print('Query success:', resp.read().decode())
except urllib.error.HTTPError as e:
    print('Query error:', e.code, e.read().decode())
