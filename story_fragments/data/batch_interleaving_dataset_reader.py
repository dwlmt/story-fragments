from typing import Dict, Mapping, Iterable
import json

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance


@DatasetReader.register("interleaving-batch")
class InterleavingBatchDatasetReader(DatasetReader):
    """
    A `DatasetReader` that wraps multiple other dataset readers,
    and interleaves their instances, adding a `MetadataField` to
    indicate the provenance of each instance.
    Unlike most of our other dataset readers, here the `file_path` passed into
    `read()` should be a JSON-serialized dictionary with one file_path
    per wrapped dataset reader (and with corresponding keys).
    Registered as a `DatasetReader` with name "interleaving".
    # Parameters
    readers : `Dict[str, DatasetReader]`
        The dataset readers to wrap. The keys of this dictionary will be used
        as the values in the MetadataField indicating provenance.
    dataset_field_name : `str`, optional (default = `"dataset"`)
        The name of the MetadataField indicating which dataset an instance came from.
    scheme : `str`, optional (default = `"round_robin"`)
        Indicates how to interleave instances. Currently the two options are "round_robin",
        which repeatedly cycles through the datasets grabbing one instance from each;
        and "all_at_once", which yields all the instances from the first dataset,
        then all the instances from the second dataset, and so on. You could imagine also
        implementing some sort of over- or under-sampling, although hasn't been done.
    """

    def __init__(
        self,
        readers: Dict[str, DatasetReader],
        dataset_field_name: str = "dataset",
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._readers = readers
        self._dataset_field_name = dataset_field_name

        self.batch_size = batch_size

    def _read_batch_at_once(self, datasets: Mapping[str, Iterable[Instance]]) -> Iterable[Instance]:
        remaining = set(datasets)
        dataset_iterators = {key: iter(dataset) for key, dataset in datasets.items()}

        while remaining:
            for key, dataset in dataset_iterators.items():
                if key in remaining:
                    try:
                        for i in range(self.batch_size):
                            instance = next(dataset)
                            instance.fields[self._dataset_field_name] = MetadataField(key)
                            yield instance
                    except StopIteration:
                        remaining.remove(key)

    def _read(self, file_path: str) -> Iterable[Instance]:
        try:
            file_paths = json.loads(file_path)
        except json.JSONDecodeError:
            raise ConfigurationError(
                "the file_path for the InterleavingDatasetReader "
                "needs to be a JSON-serialized dictionary {reader_name -> file_path}"
            )

        if file_paths.keys() != self._readers.keys():
            raise ConfigurationError("mismatched keys")

        # Load datasets
        datasets = {key: reader.read(file_paths[key]) for key, reader in self._readers.items()}

        yield from self._read_batch_at_once(datasets)
        

    def text_to_instance(self) -> Instance:  # type: ignore

        raise RuntimeError("text_to_instance doesn't make sense here")