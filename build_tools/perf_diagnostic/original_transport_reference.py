"""Bounded read-only GitHub metadata transport; credential stripped on redirects."""
import json,os,urllib.request

def fetch(path):
    class SafeRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,req,fp,code,msg,headers,newurl):
            value=super().redirect_request(req,fp,code,msg,headers,newurl)
            if value is not None:value.remove_header('Authorization')
            return value
    request=urllib.request.Request('https://api.github.com/repos/ROCm/TheRock/'+path,headers={'Authorization':'Bearer '+os.environ['GITHUB_TOKEN'],'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    with urllib.request.build_opener(SafeRedirect()).open(request,timeout=120) as response:raw=response.read(16*1024*1024+1)
    if len(raw)>16*1024*1024:raise RuntimeError('Unexpected metadata size')
    return raw
def api(path):return json.loads(fetch(path))
def pages(path,key):
    rows=[]
    for page in range(1,101):
        value=api(path+f'?per_page=100&page={page}')[key];rows+=value
        if len(value)<100:return rows
    raise RuntimeError('Pagination exceeded bound')
