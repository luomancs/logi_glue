import json
import os

import datasets



_GLUE_DESCRIPTION = """\
LogiGLUE (https://arxiv.org/abs/2310.00836) is a new benchmark with a logical reasoning spaning three logical reasoning categories, including in-domain and out-of-domain datasets.
"""

_LOGIQA_DESCRIPTION = """\
LogiQA (Multiple Choice Question Answering, Liu, Jian, et al. 2020) is a QA task where each example consists of a 
passage, a question and 4 choice candidates to answer the question.
"""

_LOGIQA_CITATION = """\
@article{liu2020logiqa,
  title={Logiqa: A challenge dataset for machine reading comprehension with logical reasoning},
  author={Liu, Jian and Cui, Leyang and Liu, Hanmeng and Huang, Dandan and Wang, Yile and Zhang, Yue},
  journal={arXiv preprint arXiv:2007.08124},
  year={2020}
}"""

class LogiGlueConfig(datasets.BuilderConfig):
    """BuilderConfig for SuperGLUE."""

    def __init__(self, name, features, data_url, citation, url, label_classes=("False", "True"), **kwargs):
        """BuilderConfig for SuperGLUE.

        Args:
        features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
        data_url: `string`, url to download the zip file from.
        citation: `string`, citation for the data set.
        url: `string`, url for information about the data set.
        label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
        **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 1.0.2: Fixed non-nondeterminism in ReCoRD.
        # 1.0.1: Change from the pre-release trial version of SuperGLUE (v1.9) to
        #        the full release (v2.0).
        # 1.0.0: S3 (new shuffling, sharding and slicing mechanism).
        # 0.0.2: Initial version.
        super(LogiGlueConfig, self).__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.name = name
        self.features = features
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url


class LogiGlue(datasets.GeneratorBasedBuilder):
    """The SuperGLUE benchmark."""

    BUILDER_CONFIGS = [
        LogiGlueConfig(
            name="logiqa",
            description=_LOGIQA_DESCRIPTION,
            features=['context', 'question', 'choices', 'answer_choice', 'answer_text', 'proof', 'question_type', 'original_dataset', 'category', 'input', 'id_', 'random_10_shot', 'bm25_10_shots'],
            data_url="https://github.com/luomancs/logi_glue/tree/main/datasets/logiqa/",
            citation=_LOGIQA_CITATION,
            url="https://github.com/lgw863/LogiQA-dataset",
            label_classes=[0,1,2,3],
        ),]
    
    def _info(self):
        features = {feature: datasets.Value("string") for feature in self.config.features}
        if self.config.name.startswith("wsc"):
            features["span1_index"] = datasets.Value("int32")
            features["span2_index"] = datasets.Value("int32")
        if self.config.name == "wic":
            features["start1"] = datasets.Value("int32")
            features["start2"] = datasets.Value("int32")
            features["end1"] = datasets.Value("int32")
            features["end2"] = datasets.Value("int32")
        if self.config.name == "multirc":
            features["idx"] = dict(
                {
                    "paragraph": datasets.Value("int32"),
                    "question": datasets.Value("int32"),
                    "answer": datasets.Value("int32"),
                }
            )
        elif self.config.name == "record":
            features["idx"] = dict(
                {
                    "passage": datasets.Value("int32"),
                    "query": datasets.Value("int32"),
                }
            )
        else:
            features["idx"] = datasets.Value("int32")

        if self.config.name == "record":
            # Entities are the set of possible choices for the placeholder.
            features["entities"] = datasets.features.Sequence(datasets.Value("string"))
            # The start and end indices of paragraph text for each entity.
            features["entity_spans"] = datasets.features.Sequence(
                {
                    "text": datasets.Value("string"),
                    "start": datasets.Value("int32"),
                    "end": datasets.Value("int32"),
                }
            )
            # Answers are the subset of entities that are correct.
            features["answers"] = datasets.features.Sequence(datasets.Value("string"))
        else:
            features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)

        return datasets.DatasetInfo(
            description=_GLUE_DESCRIPTION + self.config.description,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n"
        )
    
    def _split_generators(self, dl_manager):
        urls_to_download = self.config.data_url
        # _URL = urls_to_download
        _URLS = {
            "train": os.path.join(urls_to_download, "train.json"),
            "test": os.path.join(urls_to_download, "test.json"),
        }
        downloaded_files = dl_manager.download_and_extract(_URLS)
        print("downloaded_files['test']:", downloaded_files['test'])
        print("downloaded_files['train']", downloaded_files['train'])
        # task_name = _get_task_name_from_data_url(self.config.data_url)
        # dl_dir = os.path.join(dl_dir, task_name)
        # print(dl_dir)
        # print("-"*100)
        if self.config.name in ["axb", "axg"]:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "data_file": downloaded_files['test'],
                        "split": datasets.Split.TEST,
                    },
                ),
            ]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": downloaded_files['train'],
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": downloaded_files['test'],
                    "split": datasets.Split.TEST,
                },
            ),
        ]
        
    def _generate_examples(self, data_file, split):
        print("data_file", data_file)
        print("split", split)
        print(data_file[split])
        print("="*100)
        with open(data_file, encoding="utf-8") as f:
            data = json.load(f)
            print(len(data))
            print("="*100)
            for row in data:
                example = {feature: row[feature] for feature in self.config.features}
                yield example['id_'], example
            
