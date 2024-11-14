import os.path
import random
import string

import requests
from concurrent.futures import ThreadPoolExecutor


def download_file(url: str, save_folder: str):
    try:
        with requests.get(url, stream=True) as req:
            req.raise_for_status()
            random_chars = ''.join(random.choices(string.ascii_letters, k=3))
            filename = random_chars + '_' + url.split('/')[-1]
            save_place = os.path.join(save_folder, filename)

            with open(save_place, mode='wb') as file:
                for chunk in req.iter_content(chunk_size=1024):
                    file.write(chunk)

        print(f'File {filename} downloaded!')
    except Exception as e:
        print(f'Error while downloading file from {url}: {e}')


def thread_download(urls: list, save_folder: str):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with ThreadPoolExecutor() as executor:
        executor.map(lambda url: download_file(url, save_folder), urls)


urls = [
    'https://picsum.photos/200/300.jpg',
    'https://picsum.photos/300/400.jpg',
    'https://picsum.photos/400/500.jpg',
]
save_folder = './lab1_save'

thread_download(urls, save_folder)