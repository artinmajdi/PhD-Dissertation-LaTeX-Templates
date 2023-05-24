import pyzotero
from pyzotero import zotero
import pandas as pd
import requests
import json
import re
import requests
from titlecase import titlecase
from tqdm import tqdm


class Zotero:
    def __init__(self, api_key = "DTSD9thmaTa38IoNLdSQqNWU", user_id = "8620643", library_type = "user"):
        self.api_key      = api_key
        self.user_id      = user_id
        self.library_id   = user_id
        self.library_type = library_type
        self.library = self.get_library(user_id=user_id, api_key=api_key)

    def do_update(self):
        items = self.get_items()
        self.fix_titles(items)
        print("Finished updating titles.")

    def get_library(self, user_id, api_key):
        # Initialize Zotero library
        return zotero.Zotero(user_id, 'user', api_key)

    def get_items(self, items_per_page=100, current_page=0):
        all_items = []

        while True:
            items = self.library.top(start=items_per_page * current_page, limit=items_per_page)
            if not items:
                break

            all_items.extend(items)
            current_page += 1

        return all_items

    def fix_titles(self, items):

        def update(item, new_title):
            """ Function to update the title of an item """
            item['data']['title'] = new_title
            response = self.library.update_item(item)

            # if response.status_code != 204:
            #     print(f"Failed to update item: {item['key']} - {item['data']['title']}")

        for item in tqdm(items):

            if 'title' not in item['data']: continue

            title = item['data']['title']

            # Removing curly brackets
            title_updated = re.sub('[{}]', '', title)

            # Making the titles titlecase
            title_updated = titlecase(title_updated)

            # Update the titles
            if title != title_updated:
                print(title)
                print(title_updated)
                update(item, title_updated)

    def get_doi(self, item):
        return item['data']['DOI']
