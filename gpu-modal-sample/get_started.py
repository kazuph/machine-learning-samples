import modal

stub = modal.Stub("example-get-started")

@stub.function()
def whoami():
    # linux comandでwhoamiを実行する
    import subprocess
    return subprocess.check_output(["whoami"]).decode("utf-8").strip()


@stub.local_entrypoint()
def main():
    print(whoami.local()) # kazuph
    print(whoami.remote()) # root
