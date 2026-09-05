import re

# Mocking how Starlette handles allow_origin_regex
pattern = re.compile('https://.*\.vercel\.app')
url = 'https://documind-j581l4eia-kk036.vercel.app'
print("Match:", bool(pattern.match(url)))
