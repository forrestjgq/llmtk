import sys
from huggingface_hub import snapshot_download

repos = {
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",
    "baichuan-13b": "baichuan-inc/Baichuan2-13B-Chat",
    "falcon": "tiiuae/falcon-7b",
    "chatglm3": "THUDM/chatglm3-6b"
}
def get_repo_id(repo):
    if '/' in repo:
        return repo
    if repo in repos:
        return repos[repo]
    return None

def download(repo):
    repo_id = get_repo_id(repo)
    if repo_id:
        print(f'\n\n\n>>>>>>>>>>>>downloading {repo_id}')
        while True:
            try:
                return snapshot_download(repo_id=repo_id, resume_download=True)
            except Exception as e:
                print(str(e))
                continue
    else:
        print(f'repo {repo} not defined')
                
            
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        args = repos.keys()
    for repo in args:
        path = download(repo)
        print(f'{repo} is downloaded to {path}')