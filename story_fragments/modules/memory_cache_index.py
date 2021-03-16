import copy
import logging
from collections import OrderedDict
from typing import List, Tuple, Any

import faiss
import more_itertools
import numpy as np


logger = logging.getLogger(__name__)

class OrderedDictCache:
    ''' Simple ordered dict.
    '''

    def __init__(self, capacity: int, buffer=10, lru: bool = True, callback=None):
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
    def __init__(self, embedding_dim=768, capacity: int = 9900, buffer=100, lru: bool = True, **kwargs):

        self.embedding_dim = embedding_dim
        self.id = 0

        ''' This is a hack so that memory has different id range from the knowledgebase.
        '''
        self.id_offset = int(1e9 + 1)

        self.init_index()

        def remove_from_cache(docs: List[Tuple]):
            doc_ids = [d[0] for d in docs]
            doc_ids_arr = np.asarray(doc_ids)
            self.remove_ids(doc_ids_arr)

        self.cache = OrderedDictCache(capacity=capacity, buffer=buffer, lru=lru, callback=remove_from_cache)

    def init_index(self):
        """ Initialise the Faiss index.
        """
        index = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIDMap2(index)

    def get_doc_dicts(self, doc_ids: np.ndarray) -> List[dict]:
        """ Get dictionary information from the requested docs.

        Args:
            doc_ids (ndarray):
        """
        doc_ids = doc_ids.flatten()
        docs = []
        for id in doc_ids:

            id = int(id) - self.id_offset

            doc_dict = copy.deepcopy(self.cache.get(int(id)))
            if doc_dict is None:
                doc_dict = {"id": f"{id}", "text": "", "title": "",
                            "embeddings": np.zeros(self.embedding_dim, dtype=np.float32)}
            else:
                try:
                    doc_dict['id'] = f"{int(doc_dict['id']) + self.id_offset}"
                except TypeError:
                    pass
                except ValueError:
                    pass


            docs.append(doc_dict)

        logging.debug(f"Doc Dicts: {doc_dict['id']}, {doc_dict['title']}, {doc_dict['text']}")
        return docs

    def get_doc_dict(self, doc_id: int) -> List[dict]:
        """ Get dictionary information from the requested docs.

        Args:
            doc_ids (int):
        """

        doc_id = int(doc_id) - self.id_offset


        if doc_id < 0:
            cache_size = len(self.cache)
            doc_id

        doc_dict = copy.deepcopy(self.cache.get(int(doc_id)))
        if doc_dict is None:
            doc_dict = {"id": f"{doc_id}", "text": " ", "title": " ",
                        "embeddings": np.zeros(self.embedding_dim, dtype=np.float32)}
        else:
            try:
                doc_dict['id'] = f"{int(doc_dict['id']) + self.id_offset}"
            except TypeError:
                pass
            except ValueError:
                pass

        logging.debug(f"Doc Dicts: {doc_dict['id']}, {doc_dict['title']}, {doc_dict['text']}")

        return doc_dict

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            question_hidden_states (ndarray): Question states to match against the Faiss index.
            n_docs (int): Number of docs to retrieve.

        Returns:
            :obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`: A tensor of indices of retrieved documents.
            :obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        assert len(question_hidden_states.shape) == 2

        if n_docs >= 0:
            distances, indices, = self.index.search(np.float32(question_hidden_states), n_docs)
        else:
            distances, indices, = self.random(np.float32(question_hidden_states), n_docs)


        embeddings_list = []
        for ind in indices:
            nearest = []
            for nearest_ind in ind:
                item = self.cache.get(int(nearest_ind))

                if item is not None:
                    nearest.append(item['embeddings'])
                else:
                    nearest.append(np.zeros(self.embedding_dim, dtype=np.float32))

            embeddings_list.append(nearest)

        indices_array = np.asarray(indices)
        embeddings_array = np.asarray(embeddings_list)

        indices_array += self.id_offset

        logging.debug(f"Top Docs: {indices}, {distances}")
        return indices_array, embeddings_array, distances

    def add(self, context_dicts: List[dict], context_hidden_states: np.ndarray):
        """ Add vectors and dicts to the index.
        Args:
            context_dicts (List[dict]): A list of dictionaries with the representations. Must contain id, title and text fields.
            context_hidden_states (ndarray): The ndarray is batch size * dim.
        """

        assert len(context_dicts) > 0
        assert len(context_hidden_states.shape) == 2

        context_hidden_states = np.float32(context_hidden_states)

        id_list = []
        for item, vec in zip(context_dicts, context_hidden_states):
            item['embeddings'] = vec.tolist()
            self.cache.put(self.id, item)
            id_list.append(self.id)
            self.id += 1

        ids = np.asarray(id_list)

        logger.debug(f"Add ids to Faiss: {ids}")
        self.index.add_with_ids(context_hidden_states, ids)

        ids += self.id_offset

        return ids

    def remove_ids(self, doc_ids: np.ndarray) -> List[dict]:
        """ Remove from the dictionary and the Faiss index.
        Args:
            doc_ids (ndarray): Ids to remove.
        """
        logger.debug(f"Remove from Faiss: {doc_ids}")
        self.index.remove_ids(doc_ids)

    def clear_memory(self):
        """ Clear the Faiss index.
        """
        for chunk in more_itertools.chunked(self.cache.cache.items(),100):
            ids = [c[0] for c in chunk]
            doc_ids = np.asarray(ids)
            print(f"Clear doc_ids: {doc_ids}")
            self.index.remove_ids(doc_ids)
            

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        for item in self.cache:
            yield item
