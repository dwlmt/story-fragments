from typing import List, Tuple, Any

import faiss
import numpy as np

from collections import OrderedDict

class OrderedDictCache:
    ''' Simple ordered dict.
    '''
    def __init__(self, capacity: int, buffer=10, lru: bool = True, callback = None):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.buffer = buffer
        self.lru = lru
        self.callback = callback

    def get(self, key: Any) -> Any:
        if key not in self.cache:
            return None
        else:
            if self.lru:
                self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        self.cache[key] = value

        if self.lru:
            self.cache.move_to_end(key)

        self._cleanup()

    def remove(self, key: Any) -> None:
        item = self.cache.pop(key)

        if self.callback is not None:
            self.callback([item])

    def _cleanup(self):

        if len(self.cache) >= (self.capacity + self.buffer):
            removed_list = []

            while len(self.cache) > self.capacity:
                removed = self.cache.popitem(last=False)
                removed_list.append(removed)

            if self.callback is not None:
                self.callback(removed_list)

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        for item in self.cache.items():
            yield item


class MemoryIndex:
    def __init__(self,  capacity: int = 9900, buffer=100, lru: bool = True, **kwargs):
        self.init_index()

        def remove_from_cache(items: List[Tuple]):
            print(items)

        self.cache = OrderedDictCache(capacity=capacity, buffer=buffer, lru=lru, callback=remove_from_cache )

    def init_index(self):
        """ Initialise the Faiss index.
        """
        pass#raise NotImplementedError()

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        """ Get dictionary information from the requested docs.

        Args:
            doc_ids (ndarray):
        """
        raise NotImplementedError()

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs: (int) = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            question_hidden_states (ndarray): Question states to match against the Faiss index.
            n_docs (int): Number of docs to retrieve.
        """
        raise NotImplementedError()

    def add(self, context_dicts: List[dict], context_hidden_states: np.ndarray):
        """ Add vectors and dicts to the index.
        Args:
            context_dicts (List[dict]): A list of dictionaries with the representations. Must contain id, title and text fields.
            context_hidden_states (ndarray): The ndarray is batch size * dim.
        """
        #print("Context", context_dicts, len(self.cache))
        for i, item in enumerate(context_dicts):
            self.cache.put(item["id"], item)


    def remove_ids(self, doc_ids: np.ndarray) -> List[dict]:
        """ Remove from the dictionary and the Faiss index.
        Args:
            doc_ids (ndarray): Ids to remove.
        """
        raise NotImplementedError()

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        for item in self.cache:
            yield item