import string
from random import choice

import more_itertools
import numpy
import pytest

from story_fragments.modules.memory_cache_index import OrderedDictCache, MemoryIndex


def print_callback(item_list):
    print(f"Cache callback: {item_list}")
    assert len(item_list) == 10

@pytest.fixture
def fifo_cache():
    '''Returns a cache with a capacity of 100 '''
    return OrderedDictCache(capacity=100, lru=False, callback=print_callback)

@pytest.fixture
def lru_cache():
    '''Returns a cache with a capacity of 100 '''
    return OrderedDictCache(capacity=100, lru=True, callback=print_callback)

def test_cache_non_lru(fifo_cache):
    items = [r for r in range(0,200)]
    for item in items:
        fifo_cache.put(item, item)

    assert len(fifo_cache) == 100

    should_contain = items[100:200]
    should_not_contain = items[:100]

    for item in should_contain:
        retrieved = fifo_cache.get(item)
        print(retrieved)
        assert retrieved is not None

    for item in should_not_contain:
        retrieved = fifo_cache.get(item)
        assert retrieved is None


def test_cache_lru(lru_cache):
    items = [r for r in range(0,100)]
    for item in items:
        lru_cache.put(item, item)

    for item in items[0:50]:
        retreived = lru_cache.get(item)
        print(retreived)

    extra_items  = [r for r in range(100,150)]

    for item in extra_items:
        lru_cache.put(item, item)

    assert len(lru_cache) == 100

    should_contain = items[:50]
    for item in should_contain:
        retrieved = lru_cache.get(item)
        print(item, retrieved)
        assert retrieved is not None

    should_not_contain = items[50:100]
    for item in should_not_contain:
        retrieved = lru_cache.get(item)
        print(item, retrieved)
        assert retrieved is None


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(choice(letters) for i in range(length))
    return result_str

embedding_size = 768

@pytest.fixture
def memory_index():
    '''Default memory index'''
    return MemoryIndex()

def test_memory_index(memory_index):
    items = [r for r in range(0, 10000)]
    random_vectors = []

    for item_batch in more_itertools.chunked(items, n=100):

        index_list = []
        vector_list = []
        for item in item_batch:

            item_dict = {}
            item_dict["id"] = item
            item_dict["title"] = get_random_string(10)
            item_dict["text"] = get_random_string(100)
            index_list.append(item_dict)

            ran_vec = numpy.random.rand(embedding_size)
            vector_list.append(ran_vec)

        vectors = numpy.stack(vector_list)
        random_vectors.append(vectors)
        memory_index.add(index_list, vectors)

    search_index = 0
    closest_indices = []
    for vec_batch in random_vectors:
        indices, embeddings = memory_index.get_top_docs(vec_batch)

        first_indices = [i[0] for i in indices]
        closest_indices.extend(first_indices)

        search_index += vec_batch.shape[0]

    matches = 0
    for cl, item in zip(closest_indices, items):
        print(cl, item)
        matches += int(cl == item)

    assert matches > 9750


def test_memory_index_expired(memory_index):
    items = [r for r in range(0, 20000)]
    random_vectors = []

    for item_batch in more_itertools.chunked(items, n=100):

        index_list = []
        vector_list = []
        for item in item_batch:

            item_dict = {}
            item_dict["id"] = item
            item_dict["title"] = get_random_string(10)
            item_dict["text"] = get_random_string(100)
            index_list.append(item_dict)

            ran_vec = numpy.random.rand(embedding_size)
            vector_list.append(ran_vec)

        vectors = numpy.stack(vector_list)
        random_vectors.append(vectors)
        memory_index.add(index_list, vectors)

    for vec_batch in random_vectors:
        indices, embeddings = memory_index.get_top_docs(vec_batch)

        # The first 10000 will have expired from the cache.
        for index in indices:
            for nearest in index:
                assert nearest > 10000





