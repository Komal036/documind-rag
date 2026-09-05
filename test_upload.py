import urllib.request
import json
import urllib.error

# Login to get token
data = json.dumps({'email':'kumarikomal3434@gmail.com','password':'password'}).encode()
req = urllib.request.Request('https://documind-api-4ww0.onrender.com/api/v1/auth/login', data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req)
token = json.loads(resp.read())['access_token']
print('Got token')

# Upload dummy file
boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = (
    '--' + boundary + '\r\n'
    'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
    'Content-Type: text/plain\r\n\r\n'
    'Hello world\r\n'
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
