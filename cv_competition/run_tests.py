import json
from validation import quality_validation, get_human_readable_size, time_examine
from my_model_wrapper import MyEfficientModel

    
if __name__ == '__main__':
    model = MyEfficientModel()
    quality = quality_validation(model)
    metrics = time_examine(model)
    size = get_human_readable_size(model.model_weighs_path)

    result = {**metrics, **quality, **size}
    
    with open('q_result.json', 'w') as fp:
        json.dump(result, fp)
    