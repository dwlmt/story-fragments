

CONFIG = """
{
    "dataset_reader" : {
        "type": "interleaving",
        "readers": [
        {
            "type": "writingprompts-interleaved",
            "lazy": true,
        },
        {
            "type": "wikiplots-interleaved",
            "lazy": true,
        },
        ]
    },
    "train_data_path": ["wikiplots_dummy/train", "writingprompts_dummy/train"],
    "validation_data_path":["wikiplots_dummy/validation", "writingprompts_dummy/validation"],
    "model": {
        "type": "toy-story",
    },
    "data_loader": {
        "batch_size": 8
        "batches_per_epoch": 100, 
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}
"""

