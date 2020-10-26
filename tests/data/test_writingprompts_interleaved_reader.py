from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list, logger

from story_fragments.data.writingprompts_interleaved_reader import WritingPromptsInterleavedReader


class TestWritingPromptsInterleavedReader(AllenNlpTestCase):

    def test_read_from_file(self):
        reader = WritingPromptsInterleavedReader(lazy=True)
        instances = reader.read("writingprompts_dummy/test")
        instances = ensure_list(instances)

        logger.info(instances)

        assert("Correct number of instances", len(instances) == 10000)

        for inst in instances:
            assert ("text" in inst)
            assert ("label" in inst)
