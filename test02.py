from label_studio_ml.model import LabelStudioMLBase


class DummyModel(LabelStudioMLBase):

    def __init__(self, **kwargs):
        # 继承一下LabelStudioMLBase
        super(DummyModel, self).__init__(**kwargs)

        # 然后初始化一下要用的模型即可
        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    def predict(self, tasks, **kwargs):
        """ 核心的推理函数，在这完成label studio中图片获取、模型推理和数据打包返回
        """
        predictions = []
        for task in tasks:
            predictions.append({
                'score': 0.987,  # prediction overall score, visible in the data manager columns
                'model_version': 'delorean-20151021',  # all predictions will be differentiated by model version
                'result': [{
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'choices',
                    'score': 0.5,  # per-region score, visible in the editor
                    'value': {
                        'choices': [self.labels[0]]
                    }
                }]
            })
        return predictions