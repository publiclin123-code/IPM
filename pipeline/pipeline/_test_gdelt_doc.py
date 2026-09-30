"""临时测试: GDELT DOC 2.0 API 按关键词查新闻全文"""
import json
import ssl
import urllib.parse
import urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

query = urllib.parse.quote('"Red Sea" shipping disruption')
url = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    f"?query={query}&mode=artlist&format=json"
    "&startdatetime=20231115000000&enddatetime=20231215000000&maxrecords=5"
)
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        data = json.loads(resp.read().decode())
    arts = data.get("articles", [])
    print("articles found:", len(arts))
    for a in arts[:3]:
        print("-", a.get("seendate"), "|", (a.get("title") or "")[:80], "|", (a.get("url") or "")[:60])
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200])
