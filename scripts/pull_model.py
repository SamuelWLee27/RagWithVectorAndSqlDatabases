import ollama

current_digest = ''
for progress in ollama.pull('llama2:7b', stream=True):
    status = progress.get('status', '')
    digest = progress.get('digest', '')

    # Only print update if it's a new layer or a completion message
    if digest != current_digest or 'completed' in status:
        print(status)
        current_digest = digest