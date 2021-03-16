
class DownloadGutenberg(object):

    def download(self,
                export_dir="/home/s1569885/gutenberg/",
                subjects_to_include = ["fiction", "classics", "literature", "romance", "crime", "myth",
                                       "legend", "stories", "story", "drama"]
               ):

        from pathlib import Path

        import fire
        from gutenberg.query import get_etexts
        from gutenberg.query import get_metadata
        from gutenberg.acquire import load_etext
        from gutenberg.cleanup import strip_headers

        from gutenberg.acquire import set_metadata_cache
        from gutenberg.acquire import get_metadata_cache

        cache = get_metadata_cache()
        # cache.populate()
        set_metadata_cache(cache)

        Path(f"{export_dir}").mkdir(parents=True, exist_ok=True)

        texts = get_etexts('language', ("en"))

        texts_to_download = []

        for t in texts:
            subjects = get_metadata('subject', t)
            for s in subjects:
                for inc in subjects_to_include:
                    if inc in s.lower():
                        print(f"Match subject: {t} - {s}")
                        texts_to_download.append(t)

        texts_to_download = sorted(list(set(texts_to_download)))
        print(f"All texts to download: {texts_to_download}")
        for t in texts_to_download:

            try:
                name = next(iter(get_metadata('title', t)))
                author = next(iter(get_metadata('author', t)))
                file_name = f"{t} {name} By {author}".strip().replace(" ", "_")
                print(f"Write Book: {file_name}")

                text = strip_headers(load_etext(t)).strip()

                with open(f"{export_dir}/{file_name}.txt", "w") as f:
                    f.write(text)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    fire.Fire(DownloadGutenberg)

